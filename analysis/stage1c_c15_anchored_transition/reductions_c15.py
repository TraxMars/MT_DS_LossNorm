"""FP64 sufficient statistics without materializing dense FP64 products."""
import numpy as np
from c15_common import gradient_status

def feature_statistics(g,f,m):
    gn=np.sqrt(np.einsum('cijk,cijk->c',g,g,dtype=np.float64))
    signed=np.einsum('cijk,cijk,ijk->c',g,f,m,dtype=np.float64)
    absolute=np.einsum('cijk,cijk,ijk->c',abs(g),abs(f),m,dtype=np.float64)
    return dict(inner=signed,abs_gxa=absolute,gradient_norm=gn,status=gradient_status(gn))

def feature_cosine(a,b):
    an=np.sqrt(np.einsum('cijk,cijk->c',a,a,dtype=np.float64));bn=np.sqrt(np.einsum('cijk,cijk->c',b,b,dtype=np.float64))
    valid=lambda n:np.isfinite(n)&(n>max(1e-12,1e-6*n.max(initial=0)))
    ok=valid(an)&valid(bn);dot=np.einsum('cijk,cijk->c',a,b,dtype=np.float64);cos=np.full(len(a),np.nan)
    cos[ok]=np.clip(dot[ok]/(an[ok]*bn[ok]),-1,1)
    status=np.where(~np.isfinite(an)|~np.isfinite(bn),'nonfinite',np.where((an==0)|(bn==0),'zero_gradient',np.where(ok,'valid','below_threshold')))
    return cos,an,bn,status,dot
