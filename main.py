import argparse
import datetime
import os
import os.path as osp
import shutil
import time
import numpy as np
import torch
torch.cuda.device_count()
import yaml
from munch import Munch
from softgroup.model import get_model
# from isbnet.model import ISBNet
from softgroup.util import (AverageMeter, SummaryWriter, build_optimizer, checkpoint_save,
                            collect_results_cpu, cosine_lr_after_step, get_dist_info,
                            get_max_memory, get_root_logger, init_dist, is_main_process,
                            is_multiple, is_power2, load_checkpoint)
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
from mytools.save_npy import save_npy_instance,save_gt_instances,save_pred_instances,save_npy
from mytools.visualization import visual,visual_gt,visual_pre
from softgroup.util import rle_decode,rle_encode
#from isbnet.util import rle_decode
#from spherical_mask.model import SphericalMask
from dataset.stpls3d.prepare_data_inst_instance_hongkong_calculating import get_volume1
from mytools.merge import merge

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # choose gpu id to train on the server
def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('--city', type=str, default='Wuhu', help='path to config file')
    
    parser.add_argument('--model', type=str, default='softgroup_my8')
    parser.add_argument('--config', type=str, default='', help='path to config file')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--resume', default='', type=str, help='path to resume from')
    parser.add_argument('--work_dir', type=str, help='working directory')
    parser.add_argument('--skip_validate', default=False, action='store_true', help='skip validation')
    args = parser.parse_args()
    args.work_dir = './experiment/train_model_'+args.city+'_'+args.model
    if(args.model =='softgroup'):
        args.config='./configs/softgroup/softgroup_urbanbis.yaml'
    elif(args.model =='softgroup++'):
        args.config='./configs/softgroup++/softgroup++_urbanbis_building_pre.yaml'
    else:
        if('softgroup' in args.model):
            args.config='./configs/softgroup++/softgroup++_urbanbis_building.yaml'
    if(args.model=='isbnet'):
        args.config='./configs/isbnet/isbnet_urbanbis.yaml'
    if(args.model=='spherical'):
        args.config='./configs/spherical/config.yaml'
    return args

def train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer):
    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()
    if(epoch>20):
        semantic_only=False
    else:
        semantic_only=True
    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)
    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)
        cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            loss, log_vars = model(batch,epoch, return_loss=True)
        # meter_dict
        for k, v in log_vars.items():
            if k not in meter_dict.keys():
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)

        # # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if cfg.get('clip_grad_norm', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]['lr']

        if is_multiple(i, 50):
            log_str = f'Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  '
            log_str += f'lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, '\
                f'data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}'
            for k, v in meter_dict.items():
                log_str += f', {k}: {v.val:.4f}'
            logger.info(log_str)
    writer.add_scalar('train/learning_rate', lr, epoch)
    log_str=f'Epoch [{epoch}/{cfg.epochs}] '
    for k, v in meter_dict.items():
        writer.add_scalar(f'train/{k}', v.avg, epoch)
        log_str += f' {k}: {v.val:.4f},'
    logger.info(log_str)
    if(epoch % cfg.save_freq ==0 or epoch >=cfg.epochs-5):
        checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)

def complete(pre_masks, all_point_num):
    pre_all=0.0
    for i, pre_mask in enumerate(pre_masks):
        mask = rle_decode(pre_mask['pred_mask'])
        pre_all=pre_all+ mask.sum()
    return pre_all/all_point_num

def complete_num(pre_ins, all_inst):
    all_inst = np.unique(all_inst)
    n = abs(len(pre_ins)-len(all_inst))
   
    return n


def validate(epoch, model, val_loader, cfg, logger, writer,model_name):
    if(model_name=='isbnet'):
        # from isbnet.evaluation import (ScanNetEval, evaluate_offset_mae,
        #                            evaluate_semantic_acc, evaluate_semantic_miou)
        from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
    elif(model_name=='spherical'):
        from spherical_mask.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
    elif('softgroup' in model_name):
        from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
    logger.info('Validation')
    results = []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts,all_complete,all_complete_num = [], [], [],[],[]
    scan_ids, coords = [], []
    all_panoptic_preds = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    val_set = val_loader.dataset
    eval_tasks = cfg.model.test_cfg.eval_tasks
    eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
    scannet_eval = ScanNetEval(val_set.CLASSES, eval_min_npoint)

    #print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    all_speed = 0
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            torch.cuda.synchronize()
            #time_start = time.time()
            result = model(batch,epoch)
            torch.cuda.synchronize()
            #time_end = time.time()
            #print('Speed: %.5f FPS\n' % (1/(time_end-time_start)))
            #all_speed+=1/(time_end-time_start)
            # for instans in result['pred_instances']:
            #     pred_mask = rle_decode(instans['pred_mask'])
            #     volume = get_volume1(batch['ground_xyzs'][pred_mask].numpy())
            #     instans['volume'] = volume

            # gt_sjl=[result['gt_instances']]
            # pred_insts_sjl=[result['pred_instances']]
            # avg = scannet_eval.evaluate(pred_insts_sjl, gt_sjl, logger,batch['ground_xyzs'])


            # all_complete.append(complete(result['pred_instances'],len(result['coords_float'])))
            # all_complete_num.append(complete_num(result['pred_instances'],result['gt_instances']))
            results.append(result)
            progress_bar.update(world_size)
        #print(all_speed/i)
        #print('sjl',scannet_eval.get_mean_vol())
        progress_bar.close()
        results = collect_results_cpu(results, len(val_set))
    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            if 'semantic' in eval_tasks or 'panoptic' in eval_tasks:
                all_sem_labels.append(res['semantic_labels'])
                all_inst_labels.append(res['instance_labels'])
            if 'semantic' in eval_tasks:
                all_sem_preds.append(res['semantic_preds'])
                all_offset_preds.append(res['offset_preds'])
                all_offset_labels.append(res['offset_labels'])
            if 'instance' in eval_tasks:
                #coords.append(res['xyz_pres'])
                all_pred_insts.append(res['pred_instances'])
                all_gt_insts.append(res['gt_instances'])
                
                #all_inst_labels.append(res['instance_labels'])
                #all_offset_preds.append(res['offset_preds'])
                #all_offset_labels.append(res['offset_labels'])
            if 'panoptic' in eval_tasks:
                all_panoptic_preds.append(res['panoptic_preds'])
        
        #保存
        if 'instance' in eval_tasks and epoch==200:
            save_numbuer=4
            nyu_id = val_set.NYU_ID
            cfg.out = "./experiment/test_model"
            save_npy(cfg.out, 'coords', scan_ids[0:save_numbuer], coords[0:save_numbuer])
            #save_gt_instances(cfg.out, 'gt_instance', scan_ids[0:save_numbuer], all_gt_insts[0:save_numbuer], nyu_id)
            save_pred_instances(cfg.out, 'pred_instance', scan_ids[0:save_numbuer], all_pred_insts[0:save_numbuer], nyu_id)
            visual(cfg,scan_ids[0:save_numbuer])
            #visual_gt(cfg,scan_ids[0:save_numbuer])
        
        if 'instance' in eval_tasks:
            logger.info('Evaluate instance segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            #scannet_eval = ScanNetEval(val_set.CLASSES, eval_min_npoint)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts,logger)
            writer.add_scalar('val/AP', eval_res['all_ap'], epoch)
            writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch)
            writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch)
            logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
            # mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
            #                           cfg.model.ignore_label, logger)
            #writer.add_scalar('val/Offset MAE', mae, epoch)
            #logger.info('val/Offset MAE: {:.1f}'.format(mae))
            # logger.info('Complete: {:.3f}. Complete_num: {:.3f}.'.format(
            #     sum(all_complete)/len(all_complete), sum(all_complete_num)/len(all_complete_num)))
        if 'panoptic' in eval_tasks:
            logger.info('Evaluate panoptic segmentation')
            eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
            panoptic_eval = PanopticEval(val_set.THING, val_set.STUFF, min_points=eval_min_npoint)
            eval_res = panoptic_eval.evaluate(all_panoptic_preds, all_sem_labels, all_inst_labels)
            writer.add_scalar('val/PQ', eval_res[0], epoch)
            logger.info('PQ: {:.1f}'.format(eval_res[0]))
        if 'semantic' in eval_tasks:
            logger.info('Evaluate semantic segmentation and offset MAE')
            miou = evaluate_semantic_miou(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                          logger)
            acc = evaluate_semantic_acc(all_sem_preds, all_sem_labels, cfg.model.ignore_label,
                                        logger)
            mae = evaluate_offset_mae(all_offset_preds, all_offset_labels, all_inst_labels,
                                      cfg.model.ignore_label, logger)
            writer.add_scalar('val/mIoU', miou, epoch)
            writer.add_scalar('val/Acc', acc, epoch)
            writer.add_scalar('val/Offset MAE', mae, epoch)
            logger.info('val/mIoU: {:.1f}'.format(miou))
            logger.info('val/Acc: {:.1f}'.format(acc))
            logger.info('val/Offset MAE: {:.1f}'.format(mae))
        


def cut(batch):
    size=250
    stride=125.
    cloud = batch['ground_xyzs']
    blocks = []
    limit_max = torch.amax(cloud[:, 0:3], axis=0)
    width = int(np.ceil((limit_max[0] - size) / stride)) + 1
    depth = int(np.ceil((limit_max[1] - size) / stride)) + 1
    if(width==0):
        width=1
    if(depth==0):
        depth=1
    cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]

    for (x, y) in cells:
        xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
        ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
        cond = xcond & ycond
        new_batch = handle(batch,cond)
        blocks.append(new_batch)
    return blocks
   
def handle_result(masks,name):
    predicte_ins=[]
    for item in masks:
        result= {
            'scan_id':name,
            'conf':0.9,
            'label_id':1,
            'pred_mask':rle_encode(item)
        }
        predicte_ins.append(result)
    return predicte_ins

def marge_test(epoch, model, val_loader, cfg, logger, writer,model_name):
    from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
    val_set = val_loader.dataset
    logger.info('Validation')
    results = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    eval_tasks = cfg.model.test_cfg.eval_tasks
    all_pred,all_gt=[],[]
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):
            results = []
            all_pred_insts, all_mask = [], []
            scan_ids, coords = [], []
            for batch_s in batch:
                torch.cuda.synchronize()
                result = model(batch_s,epoch)
                torch.cuda.synchronize()
                results.append(result)

            results = collect_results_cpu(results, len(batch))
            acount=0
            for res in results:
                scan_ids.append(res['scan_id'])
                if 'instance' in eval_tasks:
                    coords.append(res['xyz_pres'])
                    all_pred_insts.append(res['pred_instances'])
                    acount +=len(res['pred_instances'])
                    all_mask.append(res['conds'])
            result_ins = merge(all_pred_insts,all_mask,acount)
            #cfg.out = "./experiment/test_model"
            #save_npy_instance(cfg.out, 'coords', scan_ids[0], coords[0],result_ins)

            pred_insts_sjl = handle_result(result_ins,name=scan_ids[0])
            all_pred.append(pred_insts_sjl)
            all_gt.append(result['gt_instances'])
            progress_bar.update(world_size)
        eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
        scannet_eval = ScanNetEval(val_set.CLASSES, eval_min_npoint)
        avg = scannet_eval.evaluate(all_pred, all_gt, logger)
        progress_bar.close()

def validate_test(epoch, model, val_loader, cfg, logger, writer,model_name):
    logger.info('Validation')
    results = []
    _, world_size = get_dist_info()
    progress_bar = tqdm(total=len(val_loader) * world_size, disable=not is_main_process())
    eval_tasks = cfg.model.test_cfg.eval_tasks
    with torch.no_grad():
        model.eval()

        for i, batch in enumerate(val_loader):
            results = []
            all_pred_insts, all_mask = [], []
            scan_ids, coords = [], []
            dst = './experiment/test_model'
            have_handle=False  
            for item2 in os.listdir(dst):
                if batch[0]['scan_ids'][0] == item2.replace('_volume.txt',''):

                    have_handle=True
                    break
            if(have_handle):
                progress_bar.update(world_size)
                continue
            p=0
            for batch_s in batch:
                #print(torch.max(batch_s['coords_float'][:,0:3], dim=0).values-torch.min(batch_s['coords_float'][:,0:3], dim=0).values)
                torch.cuda.synchronize()
                result = model(batch_s,epoch)
                torch.cuda.synchronize()
                results.append(result)
                # if(p==0):
                #     break

            results = collect_results_cpu(results, len(batch))
            acount=0
            for res in results:
                scan_ids.append(res['scan_id'])
                if 'instance' in eval_tasks:
                    coords.append(res['xyz_pres'])
                    all_pred_insts.append(res['pred_instances'])
                    acount +=len(res['pred_instances'])
                    all_mask.append(res['conds'])
            print(acount)
            result_ins = merge(all_pred_insts,all_mask,acount)
            
            cfg.out = "./experiment/test_model"
            save_npy_instance(cfg.out, 'coords', scan_ids[0], coords[0],result_ins)
            #visual_pre(coords[0],all_pred_insts,cfg.out, scan_ids[0])
            progress_bar.update(world_size)
        progress_bar.close()
           

def main(args,is_test=False):
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        init_dist()
    cfg.dist = args.dist
    cfg.data.train.data_root='../Urbanbis/'+args.city+'_softgroup_building'
    cfg.data.test.data_root='../Urbanbis/'+args.city+'_softgroup_building'
    cfg.data.val.data_root='../Urbanbis/'+args.city+'_softgroup_building'

    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file)
    logger.info(f'Config:\n{cfg_txt}')
    logger.info(f'Distributed: {args.dist}')
    logger.info(f'Mix precision training: {cfg.fp16}')
    logger.info(f'city: {cfg.data.train.data_root}')
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    # model
    if(args.model=='isbnet'):
        model = ISBNet(**cfg.model).cuda()
        from isbnet.data import build_dataloader, build_dataset
    elif(args.model=='spherical'):
        model = SphericalMask(**cfg.model).cuda()
        from spherical_mask.data import build_dataloader, build_dataset
    elif('softgroup' in args.model):
        model = get_model(cfg.model,args.model)
        from softgroup.data import build_dataloader, build_dataset
    else:
        assert False
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    #val_set = build_dataset(cfg.data.val, logger)
    test_set = build_dataset(cfg.data.test, logger)
    train_loader = build_dataloader(
        train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    #val_loader = build_dataloader(val_set, training=False, dist=args.dist, **cfg.dataloader.test)
    test_loader = build_dataloader(test_set, training=False, dist=args.dist, **cfg.dataloader.test)

    # optim
    optimizer = build_optimizer(model, cfg.optimizer)

    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f'Resume from {args.resume}')
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f'Load pretrain from {cfg.pretrain}')
        load_checkpoint(cfg.pretrain, logger, model)

    # train and val
    if(is_test==True):
        # logger.info('Val')
        validate(200, model, test_loader, cfg, logger, writer,args.model)
        logger.info('Test')
        #validate_test(200, model, test_loader, cfg, logger, writer,args.model)
    else:
        logger.info('Training')
        for epoch in range(start_epoch, cfg.epochs + 1):
            #validate(epoch, model, test_loader, cfg, logger, writer,args.model)
            train(epoch, model, optimizer, scaler, train_loader, cfg, logger, writer)
            if not args.skip_validate and (is_multiple(epoch, cfg.save_freq) or is_power2(epoch)):
                # logger.info('Val')
                # validate(epoch, model, val_loader, cfg, logger, writer,args.model)
                logger.info('Test')
                validate(epoch, model, test_loader, cfg, logger, writer,args.model)
            writer.flush()

if __name__ == '__main__':
    args = get_args()
    main(args)
