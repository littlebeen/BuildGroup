# from .softgroup_2D import SoftGroup
#from .softgroup_new import SoftGroup

def get_model(model, name):
    if(model.semantic_classes==2):
        #from .previous.softgroup_building import SoftGroup
        if(name=='softgroup' or name=='softgroup++'):
            from .previous.softgroup_building import SoftGroup
        elif(name=='softgroup_2D'):
            from .softgroup_2D import SoftGroup
        elif(name=='softgroup_my2'):
            from .softgroup_2D_my2 import SoftGroup
        elif(name=='softgroup_my3'):
            from .softgroup_2D_my3 import SoftGroup
        elif(name=='softgroup_my4'):
            from .softgroup_2D_my4 import SoftGroup
        elif(name=='softgroup_my5'):
            from .softgroup_2D_my5 import SoftGroup
        elif(name=='softgroup_my6'):
            from .softgroup_2D_my6 import SoftGroup
        elif(name=='softgroup_my7'):
            from .softgroup_2D_my7 import SoftGroup
        elif(name=='softgroup_my8'):
            from .softgroup_2D_my8 import SoftGroup
        elif(name=='softgroup_edge'):
            from .softgroup_2d_my_edge import SoftGroup
        else:
            assert False
        return SoftGroup(**model).cuda()
    else:
        from .previous.softgroup_all import SoftGroup
        return SoftGroup(**model).cuda()


__all__ = ['SoftGroup']
