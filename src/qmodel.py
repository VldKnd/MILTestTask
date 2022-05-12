import torch
from torch import nn
from torch.nn.quantized import functional as qF
from torch.nn.quantized import DeQuantize

class Q(nn.Module):

    def __init__(self, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__()
        self.weight_dtype = weight_dtype
        self.activation_dtype =activation_dtype
        self._computed = False
        self.d_min = None
        self.d_max = None
        self.q_min = torch.iinfo(activation_dtype).min
        self.q_max = torch.iinfo(activation_dtype).max

        self._activation_scale = nn.Parameter(None, requires_grad=False)
        self._activation_offset = nn.Parameter(None, requires_grad=False)


    @torch.no_grad()
    def set_activation_tensor_assym_scale_offset(self, data):
        self.d_min = min(data.min(), self.d_min) if self.d_min is not None\
                     else data.min()
        self.d_max = max(data.max(), self.d_max) if self.d_max is not None\
                     else data.max()
        _activation_scale = (self.d_max - self.d_min)/(self.q_max - self.q_min)
        _activation_offset = self.q_min - (self.d_min/_activation_scale)
        self._activation_scale = nn.Parameter(
            _activation_scale,requires_grad=False
        )
        self._activation_offset = nn.Parameter(
            _activation_offset,requires_grad=False
        )

    @torch.no_grad()
    def compile(self):
        """
        Description
        """
        self._computed = True
        del self.q_min
        del self.q_max
        del self.d_max
        del self.d_min

class QInput(Q):

    def __init__(self, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__(weight_dtype, activation_dtype)
    
    def forward(self, X):
        """
        Description
        """
        if not self._computed:
            activation = X.dequantize()
            self.set_activation_tensor_assym_scale_offset(activation)

        return torch.quantize_per_tensor(
                X, self._activation_scale, self._activation_offset, self.activation_dtype
        )

class QAvgPool2d(Q):
    
    def __init__(self, AvgPool2d, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__(weight_dtype, activation_dtype)
        self.pool_kwargs = {
            "kernel_size":AvgPool2d.kernel_size,
            "stride":AvgPool2d.stride,
            "padding":AvgPool2d.padding,
        }

    def forward(self, X):
        """
        Description
        """
        return qF.avg_pool2d(X, **self.pool_kwargs)

class QLinear(Q):

    def __init__(self, Linear, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__(weight_dtype, activation_dtype)
        self.Linear = Linear

        w = Linear.weight.data

        _weight_scale, _weight_offset = get_tensor_assym_scale_offset(
            w, self.weight_dtype
        )

        self.w  = nn.Parameter(
            torch.quantize_per_tensor(
                w, _weight_scale, _weight_offset, 
                self.weight_dtype
            ),
            requires_grad=False
        )

        self.b =  nn.Parameter(
            Linear.bias.data,
            requires_grad=False,
        )

    def forward(self, X):
        """
        Description
        """

        if not self._computed:
            activation = self.Linear(X.dequantize())
            self.set_activation_tensor_assym_scale_offset(activation)
            
        out = qF.linear(
            input=X,
            weight=self.w,
            bias=self.b,
            scale=self._activation_scale,
            zero_point=self._activation_offset
        )
        
        return out

    @torch.no_grad()
    def compile(self):
        super().compile()
        del self.Linear

class QConv2d(Q):

    def __init__(self, Conv2d, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__(weight_dtype, activation_dtype)

        self.Conv2d = Conv2d
        Conv_W = Conv2d.weight.data

        _weight_scale, _weight_offset = get_tensor_assym_scale_offset(
            Conv_W, self.weight_dtype
        )

        self.w = nn.Parameter(
            torch.quantize_per_tensor(
                Conv_W, _weight_scale, _weight_offset, 
                self.weight_dtype
            ),
            requires_grad=False
        )

        self.conv_kwargs = {
            "stride":Conv2d.stride,
            "padding":Conv2d.padding,
        }

    @torch.no_grad()
    def forward(self, X):

        if not self._computed:
            self.set_activation_tensor_assym_scale_offset(X)
            
        out = qF.conv2d(
            input=X,
            weight=self.w,
            bias=None,
            scale=self._activation_scale,
            zero_point=self._activation_offset,
            dtype=self.weight_dtype,
            **self.conv_kwargs
        )
        return out

    @torch.no_grad()
    def compile(self):
        super().compile()
        del self.Conv2d

class QConvBatch2d(QConv2d):

    def __init__(self, Conv2d, BatchNorm, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__(Conv2d, weight_dtype, activation_dtype)
        self.Conv2d = Conv2d
        self.BatchNorm = BatchNorm
        conv_W = Conv2d.weight.data
        bn_W = BatchNorm.weight.data
        bn_var = BatchNorm.running_var.data
        bn_b = BatchNorm.bias.data
        bn_mean = BatchNorm.running_mean.data
        bn_eps = BatchNorm.eps

        w = (bn_W[:, None, None, None]*conv_W)/torch.sqrt(bn_var[:, None, None, None] + bn_eps)
        
        _weight_scale, _weight_offset = get_tensor_assym_scale_offset(
            w, self.weight_dtype
        )

        self.w  = nn.Parameter(
            torch.quantize_per_tensor(
                w, _weight_scale, _weight_offset, 
                self.weight_dtype
            ),
            requires_grad=False
        )
        
        self.b =  nn.Parameter(
            bn_b - (bn_W*bn_mean)/torch.sqrt(bn_var + bn_eps),
            requires_grad=False,
        )


    @torch.no_grad()
    def forward(self, X):

        if not self._computed:
            activation = self.BatchNorm(self.Conv2d(X.dequantize()))
            self.set_activation_tensor_assym_scale_offset(activation)
            
        out = qF.conv2d(
            input=X,
            weight=self.w,
            bias=self.b,
            scale=self._activation_scale,
            zero_point=self._activation_offset,
            dtype=self.weight_dtype,
            **self.conv_kwargs
        )
        return out

    @torch.no_grad()
    def compile(self):
        super().compile()
        del self.BatchNorm

class QConvBatchReLU2d(QConvBatch2d):

    def __init__(self, Conv2d, BatchNorm, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__(Conv2d, BatchNorm, weight_dtype, activation_dtype)

    @torch.no_grad()
    def forward(self, X):
        return torch.nn.functional.relu(super().forward(X))
        
class QSkipConnection(nn.Module):

    def __init__(self, f_m, f_s=None):
        """
        Description
        """
        super().__init__()
        self.f_m = f_m
        self.f_s = f_s
        self.q_add = torch.nn.quantized.QFunctional().add
        
    def forward(self, X):
        """
        Description
        """
        if self.f_s is not None:
            return torch.relu(self.q_add(self.f_s(X),self.f_m(X)))
        else:
            return torch.relu(self.q_add(X,self.f_m(X)))

class QBlockConnection(nn.Module):

    def __init__(self, BlockConnection, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__()
        self.connections = nn.ModuleList([
            QSkipConnection(
                nn.Sequential(
                    QConvBatchReLU2d(Connection.f_m[0], Connection.f_m[1], weight_dtype, activation_dtype),
                    QConvBatch2d(Connection.f_m[3], Connection.f_m[4], weight_dtype, activation_dtype)
                ),
            ) for Connection in BlockConnection.connections])

    def forward(self, X):
        """
        Description
        """
        out = X
        for i, module in enumerate(self.connections):
            out = module(out)

        return out

class QDownsamplingConnection(nn.Module):

    def __init__(self, DownsamplingConnection, weight_dtype=torch.qint8, activation_dtype=torch.quint8):
        """
        Description
        """
        super().__init__()
        self.module =  QSkipConnection(
            nn.Sequential(
                QConvBatchReLU2d(DownsamplingConnection.module.f_m[0], 
                                DownsamplingConnection.module.f_m[1], weight_dtype, activation_dtype),
                QConvBatch2d(DownsamplingConnection.module.f_m[3], 
                            DownsamplingConnection.module.f_m[4], weight_dtype, activation_dtype),
            ),
            nn.Sequential(
                QConvBatchReLU2d(DownsamplingConnection.module.f_s[0], 
                                DownsamplingConnection.module.f_s[1], weight_dtype, activation_dtype),
            ),
        )

    def forward(self, X):
        """
        Description
        """
        return self.module(X)
 
class QFlatten(nn.Module):

    def __init__(self, Flatten):
        """
        Description
        """
        super().__init__()
        
    def forward(self, X):
        return torch.flatten(X, 1, -1)

def compile_module(m):
    if hasattr(m, "compile"):
        m.compile()

map_to_q = {
    "Conv2d":QConv2d,
    "ConvBatch2d":QConvBatch2d,
    "ConvBatchReLU2d":QConvBatchReLU2d,
    "BlockConnection":QBlockConnection,
    "DownsamplingConnection":QDownsamplingConnection,
    "AvgPool2d":QAvgPool2d,
    "Flatten":QFlatten,
    "Linear":QLinear
}

@torch.no_grad()
def get_tensor_assym_scale_offset(data, dtype):
    d_min = data.min()
    d_max = data.max()
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    scale = (d_max - d_min)/(q_max - q_min)
    offset = q_min - (d_min/scale)
    return scale, offset

def quantize_model(model):
    qmodels = [QInput()]
    k, n_modules = 0, len(model)
    while k < n_modules:
        if model[k]._get_name() == "Conv2d":

            if model[k+2]._get_name() == "ReLU" and\
               model[k+1]._get_name() == "BatchNorm2d":
                qmodels.append(
                    map_to_q["ConvBatchReLU2d"](model[k], model[k+1])
                )
                k+=3

            elif model[k+1]._get_name() == "BatchNorm2d":
                qmodels.append(
                    map_to_q["ConvBatch2d"](model[k], model[k+1])
                    )
                k+=2
        else:
            qmodels.append(map_to_q[model[k]._get_name()](model[k]))
            k+=1

    qmodels.append(DeQuantize())
    return nn.Sequential(*qmodels)

def quantize_merge_model(model):
    qmodels = [QInput()]
    k, n_modules = 0, len(model)
    while k < n_modules:
        if model[k]._get_name() == "Conv2d":

            if model[k+2]._get_name() == "ReLU" and\
               model[k+1]._get_name() == "BatchNorm2d":
                qmodels.append(
                    map_to_q["ConvBatchReLU2d"](model[k], model[k+1])
                )
                k+=3

            elif model[k+1]._get_name() == "BatchNorm2d":
                qmodels.append(
                    map_to_q["ConvBatch2d"](model[k], model[k+1])
                    )
                k+=2
        else:
            qmodels.append(map_to_q[model[k]._get_name()](model[k]))
            k+=1

    qmodels.append(DeQuantize())
    return nn.Sequential(*qmodels)