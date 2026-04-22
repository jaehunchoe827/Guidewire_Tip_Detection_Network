import math
import torch
import copy
from gwtd.utils.util import make_anchors, load_weight

def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation, k=1, s=1, p=0, g=1, use_norm = True):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        self.use_norm = use_norm
        # here the eps and momentum are set to its default values
        # pytorch defaults: eps=1e-5, momentum=0.1
        # in jahongir7174's code, the eps and momentum are set to 1e-3 and 0.03, respectively
        self.norm = torch.nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.1)
        self.relu = activation

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        return self.relu(x)

    def fuse_forward(self, x):
        return self.relu(self.conv(x))


class Residual(torch.nn.Module):
    def __init__(self, ch, e=0.5):
        super().__init__()
        self.conv1 = Conv(ch, int(ch * e), torch.nn.SiLU(), k=3, p=1)
        self.conv2 = Conv(int(ch * e), ch, torch.nn.SiLU(), k=3, p=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CSPModule(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch, out_ch // 2, torch.nn.SiLU())
        self.conv3 = Conv(2 * (out_ch // 2), out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(Residual(out_ch // 2, e=1.0),
                                         Residual(out_ch // 2, e=1.0))

    def forward(self, x):
        y = self.res_m(self.conv1(x))
        return self.conv3(torch.cat((y, self.conv2(x)), dim=1))


class CSP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, n, csp, r):
        super().__init__()
        self.conv1 = Conv(in_ch, 2 * (out_ch // r), torch.nn.SiLU())
        self.conv2 = Conv((2 + n) * (out_ch // r), out_ch, torch.nn.SiLU())

        if not csp:
            self.res_m = torch.nn.ModuleList(Residual(out_ch // r) for _ in range(n))
        else:
            self.res_m = torch.nn.ModuleList(CSPModule(out_ch // r, out_ch // r) for _ in range(n))

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.res_m)
        return self.conv2(torch.cat(y, dim=1))


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2, torch.nn.SiLU())
        self.conv2 = Conv(in_ch * 2, out_ch, torch.nn.SiLU())
        self.res_m = torch.nn.MaxPool2d(k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat(tensors=[x, y1, y2, self.res_m(y2)], dim=1))


class Attention(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.num_head = num_head
        self.dim_head = ch // num_head
        self.dim_key = self.dim_head // 2
        self.scale = self.dim_key ** -0.5

        self.qkv = Conv(ch, ch + self.dim_key * num_head * 2, torch.nn.Identity())

        self.conv1 = Conv(ch, ch, torch.nn.Identity(), k=3, p=1, g=ch)
        self.conv2 = Conv(ch, ch, torch.nn.Identity())

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(b, self.num_head, self.dim_key * 2 + self.dim_head, h * w)

        q, k, v = qkv.split([self.dim_key, self.dim_key, self.dim_head], dim=2)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.conv1(v.reshape(b, c, h, w))
        return self.conv2(x)


class PSABlock(torch.nn.Module):

    def __init__(self, ch, num_head):
        super().__init__()
        self.conv1 = Attention(ch, num_head)
        self.conv2 = torch.nn.Sequential(Conv(ch, ch * 2, torch.nn.SiLU()),
                                         Conv(ch * 2, ch, torch.nn.Identity()))

    def forward(self, x):
        x = x + self.conv1(x)
        return x + self.conv2(x)


class PSA(torch.nn.Module):
    def __init__(self, ch, n):
        super().__init__()
        self.conv1 = Conv(ch, 2 * (ch // 2), torch.nn.SiLU())
        self.conv2 = Conv(2 * (ch // 2), ch, torch.nn.SiLU())
        self.res_m = torch.nn.Sequential(*(PSABlock(ch // 2, ch // 128) for _ in range(n)))

    def forward(self, x):
        x, y = self.conv1(x).chunk(2, 1)
        return self.conv2(torch.cat(tensors=(x, self.res_m(y)), dim=1))


class DarkNet(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(CSP(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(CSP(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(CSP(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(CSP(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPP(width[5], width[5]))
        self.p5.append(PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class DarkFPN(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2)
        self.h1 = CSP(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = CSP(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = CSP(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = CSP(width[4] + width[5], width[5], depth[5], csp[1], r=2)

    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))
        return p3, p4, p5


class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                           Conv(x, cls, torch.nn.SiLU()),
                                                           Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                           Conv(cls, cls, torch.nn.SiLU()),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class YOLO(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self


def yolo_v11_n(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_s(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_m(num_classes: int = 80):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_l(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(width, depth, csp, num_classes)


def yolo_v11_x(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return YOLO(width, depth, csp, num_classes)


class SobelMag(torch.nn.Module):
    # Fixed 1-ch gradient magnitude (for edge-assisted refinement).
    def __init__(self):
        super().__init__()
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        gy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('gx', gx, persistent=False)
        self.register_buffer('gy', gy, persistent=False)
    def forward(self, img):  # img: (batch_size, 3, height, width)
        x = img.mean(1, keepdim=True)  # to gray
        gx = torch.nn.functional.conv2d(x, self.gx, padding=1)
        gy = torch.nn.functional.conv2d(x, self.gy, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
        # normalize lightly to [0,1]
        mag = mag / (mag.amax(dim=[2,3], keepdim=True) + 1e-6)
        return mag  # (batch_size, 1, height, width)


class DSConv(torch.nn.Module):
    # Depthwise Separable Convolution
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = Conv(in_ch, in_ch, torch.nn.SiLU(), k=3, s=1, p=1, g=in_ch)
        self.pointwise = Conv(in_ch, out_ch, torch.nn.SiLU(), k=1, s=1, p=0, g=1)
    
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class SE(torch.nn.Module):
    # Squeeze-Excite (optional, cheap channel attention).
    def __init__(self, c: int, r: int = 16):
        super().__init__()
        self.avg = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(c, max(4, c // r), 1),
            torch.nn.SiLU(),
            torch.nn.Conv2d(max(4, c // r), c, 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w



class GuidewireDetectionHeadVer4(torch.nn.Module):
    def __init__(self, input_image_shape: tuple, feature_channels: list, config_head: dict, from_logits: bool = True):
        super().__init__()
        self.nhc = config_head['num_hidden_channels'] # hidden channels
        self.input_image_shape = input_image_shape
        self.feature_channels = feature_channels
        self.c3 = feature_channels[0]
        self.c4 = feature_channels[1]
        self.c5 = feature_channels[2]
        self.use_se = config_head['use_se']
        self.edge_assist = config_head['edge_assist']
        # image feature extraction
        self.conv_image = torch.nn.Sequential(
            Conv(3, 32, torch.nn.SiLU(), k=3, s=2, p=1),
            Conv(32, 64, torch.nn.SiLU(), k=3, s=2, p=1),
            Conv(64, self.nhc, torch.nn.SiLU(), k=3, s=1, p=1),
        )
        # lateral
        self.conv_l3 = torch.nn.Sequential(
            Conv(self.c3, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        self.conv_l4 = torch.nn.Sequential(
            Conv(self.c4, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        self.conv_l5 = torch.nn.Sequential(
            Conv(self.c5, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        self.se3 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        self.se4 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        self.se5 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        # fuse features
        self.fuse160 = torch.nn.Sequential(
            Conv(4*self.nhc, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        # decoder
        self.dec320 = DSConv(self.nhc, self.nhc)
        self.dec640 = DSConv(self.nhc, self.nhc)
        if self.edge_assist:
            self.sobel = SobelMag()
            self.edge_refine = torch.nn.Sequential(
                Conv(self.nhc+1, self.nhc, torch.nn.SiLU(), k=3, p=1),
                DSConv(self.nhc, self.nhc),
            )
        # conv for output
        self.conv_output = DSConv(self.nhc, 64)
        # final prediction
        if from_logits:
            self.pred = Conv(64, 1, torch.nn.Identity(), k=1, p=0, use_norm=False)
        else:
            self.pred = Conv(64, 1, torch.nn.Sigmoid(), k=1, p=0, use_norm=False)

    def forward(self, x):
        # here the input is list of 4 tensors
        # [input image, feature from shallow layer, feature from middle layer, feature from deep layer]
        x_image = x[0] # input image. shape: (batch_size, 3, 640, 640)
        p3 = x[1]
        p4 = x[2]
        p5 = x[3]
        # image feature extraction
        imf = self.conv_image(x_image) # (batch_size, nhc, 160, 160)
        # lateral connetctions
        l3 = self.se3(self.conv_l3(p3)) # (batch_size, nhc, 80, 80)
        l4 = self.se4(self.conv_l4(p4)) # (batch_size, nhc, 40, 40)
        l5 = self.se5(self.conv_l5(p5)) # (batch_size, nhc, 20, 20)
        # upsample l3, l4, l5 to 160x160
        l3 = self.up_sample(l3, scale_factor=2)
        l4 = self.up_sample(l4, scale_factor=4)
        l5 = self.up_sample(l5, scale_factor=8)
        # concatenate i3, l3, l4, l5
        features = [imf, l3, l4, l5]
        features = torch.cat(features, dim=1)
        # decode features
        h160 = self.fuse160(features)
        h320 = self.dec320(self.up_sample(h160))
        h640 = self.dec640(self.up_sample(h320))
        if self.edge_assist:
            edge = self.sobel(x_image)
            h640 = self.edge_refine(torch.cat([h640, edge], dim=1))
        out = self.conv_output(h640)
        out = self.pred(out)
        # (b, 1, h, w) -> (b, h, w)
        out = out.squeeze(1)
        return out

    def up_sample(self, x, scale_factor=2): # bilinear upsample
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')

    def down_sample(self, x): # max pooling downsample
        return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)


class GuidewireDetectionHeadVer3(torch.nn.Module):
    def __init__(self, input_image_shape: tuple, feature_channels: list, config_head: dict, from_logits: bool = True):
        super().__init__()
        self.nhc = config_head['num_hidden_channels'] # hidden channels
        self.input_image_shape = input_image_shape
        self.feature_channels = feature_channels
        self.c3 = feature_channels[0]
        self.c4 = feature_channels[1]
        self.c5 = feature_channels[2]
        self.use_se = config_head['use_se']
        self.edge_assist = config_head['edge_assist']
        # image feature extraction
        self.conv_image = torch.nn.Sequential(
            Conv(3, 32, torch.nn.SiLU(), k=3, s=2, p=1),
            Conv(32, 64, torch.nn.SiLU(), k=3, s=2, p=1),
            Conv(64, self.nhc, torch.nn.SiLU(), k=3, s=1, p=1),
        )
        # lateral
        self.conv_l3 = torch.nn.Sequential(
            Conv(self.c3, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        self.conv_l4 = torch.nn.Sequential(
            Conv(self.c4, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        self.conv_l5 = torch.nn.Sequential(
            Conv(self.c5, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        self.se3 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        self.se4 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        self.se5 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        # fuse features
        self.fuse160 = torch.nn.Sequential(
            Conv(4*self.nhc, self.nhc, torch.nn.SiLU(), k=1, p=0),
            DSConv(self.nhc, self.nhc),
        )
        # decoder
        self.dec320 = DSConv(self.nhc, self.nhc)
        self.dec640 = DSConv(self.nhc, self.nhc)
        if self.edge_assist:
            self.sobel = SobelMag()
            self.edge_refine = DSConv(self.nhc+1, self.nhc)
        # conv for output
        self.conv_output = DSConv(self.nhc, 64)
        # final prediction
        if from_logits:
            self.pred = Conv(64, 1, torch.nn.Identity(), k=1, p=0, use_norm=False)
        else:
            self.pred = Conv(64, 1, torch.nn.Sigmoid(), k=1, p=0, use_norm=False)

    def forward(self, x):
        # here the input is list of 4 tensors
        # [input image, feature from shallow layer, feature from middle layer, feature from deep layer]
        x_image = x[0] # input image. shape: (batch_size, 3, 640, 640)
        p3 = x[1]
        p4 = x[2]
        p5 = x[3]
        # image feature extraction
        imf = self.conv_image(x_image) # (batch_size, nhc, 160, 160)
        # lateral connetctions
        l3 = self.se3(self.conv_l3(p3)) # (batch_size, nhc, 80, 80)
        l4 = self.se4(self.conv_l4(p4)) # (batch_size, nhc, 40, 40)
        l5 = self.se5(self.conv_l5(p5)) # (batch_size, nhc, 20, 20)
        # upsample l3, l4, l5 to 160x160
        l3 = self.up_sample(l3, scale_factor=2)
        l4 = self.up_sample(l4, scale_factor=4)
        l5 = self.up_sample(l5, scale_factor=8)
        # concatenate i3, l3, l4, l5
        features = [imf, l3, l4, l5]
        features = torch.cat(features, dim=1)
        # decode features
        h160 = self.fuse160(features)
        h320 = self.dec320(self.up_sample(h160))
        h640 = self.dec640(self.up_sample(h320))
        if self.edge_assist:
            edge = self.sobel(x_image)
            h640 = self.edge_refine(torch.cat([h640, edge], dim=1))
        out = self.conv_output(h640)
        out = self.pred(out)
        # (b, 1, h, w) -> (b, h, w)
        out = out.squeeze(1)
        return out

    def up_sample(self, x, scale_factor=2): # bilinear upsample
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')

    def down_sample(self, x): # max pooling downsample
        return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)


class GuidewireDetectionHeadVer2(torch.nn.Module):
    def __init__(self, input_image_shape: tuple, feature_channels: list, config_head: dict, from_logits: bool = True):
        super().__init__()
        self.nhc = config_head['num_hidden_channels'] # hidden channels
        self.input_image_shape = input_image_shape
        self.feature_channels = feature_channels
        self.c3 = feature_channels[0]
        self.c4 = feature_channels[1]
        self.c5 = feature_channels[2]
        self.use_se = config_head['use_se']
        # image feature extraction
        self.conv_image_1 = Conv(3, 32, torch.nn.SiLU(), k=3, s=2, p=1) # (3, 640, 640) -> (32, 320, 320)
        self.conv_image_2 = Conv(32, 64, torch.nn.SiLU(), k=3, s=2, p=1) # (32, 320, 320) -> (64, 160, 160)
        self.conv_image_3 = Conv(64, 64, torch.nn.SiLU(), k=3, s=1, p=1) # (64, 160, 160) -> (64, 160, 160)
        # lateral
        self.conv_l3 = Conv(self.c3, self.nhc, torch.nn.SiLU(), k=1, p=0)
        self.conv_l3_2 = Conv(self.nhc, self.nhc, torch.nn.SiLU(), k=3, p=1)
        self.conv_l4 = Conv(self.c4, self.nhc, torch.nn.SiLU(), k=1, p=0)
        self.conv_l4_2 = Conv(self.nhc, self.nhc, torch.nn.SiLU(), k=3, p=1)
        self.conv_l5 = Conv(self.c5, self.nhc, torch.nn.SiLU(), k=1, p=0)
        self.conv_l5_2 = Conv(self.nhc, self.nhc, torch.nn.SiLU(), k=3, p=1)
        self.se3 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        self.se4 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        self.se5 = SE(self.nhc) if self.use_se else torch.nn.Identity()
        # top-down
        self.conv_td3 = Conv(2*self.nhc, self.nhc, torch.nn.SiLU(), k=3, p=1)
        self.conv_td4 = Conv(2*self.nhc, self.nhc, torch.nn.SiLU(), k=3, p=1)
        # botton-up pan
        self.conv_bu4 = Conv(2*self.nhc, self.nhc, torch.nn.SiLU(), k=3, p=1)
        # decoder
        self.dec160 = DSConv(64+2*self.nhc, self.nhc)
        self.dec320 = DSConv(self.nhc, 64)
        self.dec640 = DSConv(64, 64)
        # conv for output
        self.conv_output = Conv(64, 32, torch.nn.SiLU(), k=3, p=1)
        # final prediction
        if from_logits:
            self.pred = Conv(32, 1, torch.nn.Identity(), k=1, p=0, use_norm=False)
        else:
            self.pred = Conv(32, 1, torch.nn.Sigmoid(), k=1, p=0, use_norm=False)

    def forward(self, x):
        # here the input is list of 4 tensors
        # [input image, feature from shallow layer, feature from middle layer, feature from deep layer]
        x_image = x[0] # input image. shape: (batch_size, 3, 640, 640)
        p3 = x[1]
        p4 = x[2]
        p5 = x[3]
        # image feature extraction
        i1 = self.conv_image_1(x_image) # (batch_size, 32, 320, 320)
        i2 = self.conv_image_2(i1) # (batch_size, 64, 160, 160)
        i3 = self.conv_image_3(i2) # (batch_size, 64, 160, 160)
        # lateral connetctions
        l3 = self.se3(self.conv_l3_2(self.conv_l3(p3))) # (batch_size, nhc, 80, 80)
        l4 = self.se4(self.conv_l4_2(self.conv_l4(p4))) # (batch_size, nhc, 40, 40)
        l5 = self.se5(self.conv_l5_2(self.conv_l5(p5))) # (batch_size, nhc, 20, 20)
        # top-down connections
        f4 = self.conv_td4(torch.cat([self.up_sample(l5), l4], dim=1)) # (batch_size, nhc, 40, 40)
        f3 = self.conv_td3(torch.cat([self.up_sample(f4), l3], dim=1)) # (batch_size, nhc, 80, 80)
        # botton-up pan
        g4 = self.conv_bu4(torch.cat([self.down_sample(f3), f4], dim=1)) # (batch_size, nhc, 40, 40)
        # decoder
        h80 = f3
        features = [self.up_sample(h80),
                    self.up_sample(g4, scale_factor=4),
                    i3]
        h160 = self.dec160(torch.cat(features, dim=1))
        h320 = self.dec320(self.up_sample(h160))
        h640 = self.dec640(self.up_sample(h320))
        h640 = self.conv_output(h640)
        out = self.pred(h640)
        # (b, 1, h, w) -> (b, h, w)
        out = out.squeeze(1)
        return out

    def up_sample(self, x, scale_factor=2): # bilinear upsample
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode='bilinear')

    def down_sample(self, x): # max pooling downsample
        return torch.nn.functional.max_pool2d(x, kernel_size=2, stride=2)


class GuidewireDetectionHeadVer1(torch.nn.Module):
    def __init__(self, input_image_shape: tuple, feature_channels: list, config_head: dict, from_logits: bool = True):
        super().__init__()
        n_hidden_channels = config_head['num_hidden_channels']
        self.input_image_shape = input_image_shape
        self.feature_channels = feature_channels
        self.num_convs_for_input_feature = config_head['num_convs_for_input_feature']
        self.num_convs_for_output = config_head['num_convs_for_output']
        # input feature
        self.convs_feature_input = torch.nn.ModuleList([Conv(3, n_hidden_channels, torch.nn.SiLU(), k=3, p=1)])
        for _ in range(self.num_convs_for_input_feature-1):
            self.convs_feature_input.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # shallow feature
        self.upsamples_feature_1 = torch.nn.ModuleList([torch.nn.Upsample(scale_factor=2, mode='bilinear')]) # (80, 80) -> (160, 160)
        self.upsamples_feature_1.append(Conv(feature_channels[0], n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        for _ in range (2): # (160, 160) -> (320, 320) -> (640, 640)
            self.upsamples_feature_1.append(torch.nn.Upsample(scale_factor=2, mode='bilinear')) 
            self.upsamples_feature_1.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # middle feature
        self.upsamples_feature_2 = torch.nn.ModuleList([torch.nn.Upsample(scale_factor=2, mode='bilinear')]) # (40, 40) -> (80, 80)
        self.upsamples_feature_2.append(Conv(feature_channels[1], n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        for _ in range (3): # (80, 80) -> (160, 160) -> (320, 320) -> (640, 640)
            self.upsamples_feature_2.append(torch.nn.Upsample(scale_factor=2, mode='bilinear')) 
            self.upsamples_feature_2.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # deep feature
        self.upsamples_feature_3 = torch.nn.ModuleList([torch.nn.Upsample(scale_factor=2, mode='bilinear')]) # (20, 20) -> (40, 40)
        self.upsamples_feature_3.append(Conv(feature_channels[2], n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        for _ in range (4): # (40, 40) -> (80, 80) -> (160, 160) -> (320, 320) -> (640, 640)
            self.upsamples_feature_3.append(torch.nn.Upsample(scale_factor=2, mode='bilinear')) 
            self.upsamples_feature_3.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # output conv layers
        self.convs_output = torch.nn.ModuleList([Conv(4*n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=1, p=0)])
        for _ in range(self.num_convs_for_output-2):
            self.convs_output.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        if from_logits:
            self.convs_output.append(Conv(n_hidden_channels, 1, torch.nn.Identity(), k=3, p=1, use_norm=False))
        else:
            self.convs_output.append(Conv(n_hidden_channels, 1, torch.nn.Sigmoid(), k=3, p=1, use_norm=False))

    def forward(self, x):
        # here the input is list of 4 tensors
        # [input image, feature from shallow layer, feature from middle layer, feature from deep layer]
        x0 = x[0] # input image. shape: (batch_size, 3, 640, 640)
        for conv in self.convs_feature_input:
            x0 = conv(x0)
        x1 = x[1] # feature from shallow layer. shape: (batch_size, n_hidden_channels, 80, 80)
        for layer in self.upsamples_feature_1:
            x1 = layer(x1)
        x2 = x[2] # feature from middle layer. shape: (batch_size, n_hidden_channels, 40, 40)
        for layer in self.upsamples_feature_2:
            x2 = layer(x2)
        x3 = x[3] # feature from deep layer. shape: (batch_size, n_hidden_channels, 20, 20)
        for layer in self.upsamples_feature_3:
            x3 = layer(x3)
        # concatenate input, shallow, middle, and deep features
        x = torch.cat([x0, x1, x2, x3], dim=1)
        # pass the concatenated features to the output conv layers
        for layer in self.convs_output:
            x = layer(x)
        # (batch_size, 1, input_image_shape[0], input_image_shape[1]) ->
        # (batch_size, input_image_shape[0], input_image_shape[1])
        x = x.squeeze(1)
        return x


class GuidewireDetectionHeadVer0(torch.nn.Module):
    def __init__(self, input_image_shape: tuple, feature_channels: list, config_head: dict, from_logits: bool = True):
        super().__init__()
        n_hidden_channels = config_head['num_hidden_channels']
        self.input_image_shape = input_image_shape
        self.feature_channels = feature_channels
        self.num_convs_for_input_feature = config_head['num_convs_for_input_feature']
        self.num_convs_for_shallow_feature = config_head['num_convs_for_shallow_feature']
        self.num_convs_for_middle_feature = config_head['num_convs_for_middle_feature']
        self.num_convs_for_deep_feature = config_head['num_convs_for_deep_feature']
        self.num_convs_for_merged_feature = config_head['num_convs_for_merged_feature']
        # input feature
        self.convs_feature_input = torch.nn.ModuleList([Conv(3, n_hidden_channels, torch.nn.SiLU(), k=3, p=1)])
        for _ in range(self.num_convs_for_input_feature-1):
            self.convs_feature_input.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # shallow feature
        self.convs_feature_1 = torch.nn.ModuleList([Conv(feature_channels[0], n_hidden_channels, torch.nn.SiLU(), k=3, p=1)])
        for _ in range(self.num_convs_for_shallow_feature-1):
            self.convs_feature_1.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # middle feature
        self.convs_feature_2 = torch.nn.ModuleList([Conv(feature_channels[1], n_hidden_channels, torch.nn.SiLU(), k=3, p=1)])
        for _ in range(self.num_convs_for_middle_feature-1):
            self.convs_feature_2.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # deep feature
        self.convs_feature_3 = torch.nn.ModuleList([Conv(feature_channels[2], n_hidden_channels, torch.nn.SiLU(), k=3, p=1)])
        for _ in range(self.num_convs_for_deep_feature-1):
            self.convs_feature_3.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        # merged feature
        self.convs_output = torch.nn.ModuleList([Conv(4*n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1)])
        for _ in range(self.num_convs_for_merged_feature-2):
            self.convs_output.append(Conv(n_hidden_channels, n_hidden_channels, torch.nn.SiLU(), k=3, p=1))
        if from_logits:
            self.convs_output.append(Conv(n_hidden_channels, 1, torch.nn.Identity(), k=3, p=1, use_norm=False))
        else:
            self.convs_output.append(Conv(n_hidden_channels, 1, torch.nn.Sigmoid(), k=3, p=1, use_norm=False))
        self.upsample2 = torch.nn.Upsample(scale_factor=2)
        self.upsample3 = torch.nn.Upsample(scale_factor=4)

    def forward(self, x):
        # here the input is list of 4 tensors
        # [input image, feature from shallow layer, feature from middle layer, feature from deep layer]
        x0 = x[0] # input image. shape: (batch_size, 3, height, width)
        for conv in self.convs_feature_input:
            x0 = conv(x0)
        x1 = x[1] # feature from shallow layer. shape: (batch_size, n_hidden_channels, 80, 80)
        for conv in self.convs_feature_1:
            x1 = conv(x1)
        x2 = x[2] # feature from middle layer. shape: (batch_size, n_hidden_channels, 40, 40)
        for conv in self.convs_feature_2:
            x2 = conv(x2)
        x3 = x[3] # feature from deep layer. shape: (batch_size, n_hidden_channels, 20, 20)
        for conv in self.convs_feature_3:
            x3 = conv(x3)
        # upsample the deep and middle layers
        x2 = self.upsample2(x2)
        x3 = self.upsample3(x3)
        # concatenate the features
        x = torch.cat((x1, x2, x3), dim=1) # (batch_size, 3*n_hidden_channels, 80, 80)
        # resize the x to (batch_size, 3*n_hidden_channels, input_image_shape[0], input_image_shape[1])
        x = torch.nn.functional.interpolate(x,
                                            size=(self.input_image_shape[0], self.input_image_shape[1]),
                                            mode='bilinear', align_corners=False)
        # concatenate x0 and x
        x = torch.cat((x0, x), dim=1)
        for conv in self.convs_output:
            x = conv(x)
        # (batch_size, 1, input_image_shape[0], input_image_shape[1]) ->
        # (batch_size, input_image_shape[0], input_image_shape[1])
        x = x.squeeze(1)
        return x

class YOLOwithCustomHead(torch.nn.Module):
    def __init__(self, name_of_yolo_model: str, pretrained_weights_path: str,
                 input_image_shape: tuple, config_head: dict, from_logits: bool = True):
        super().__init__()
        # load the yolo model
        if name_of_yolo_model == 'yolo11n':
            src_yolo_model = yolo_v11_n()
            feature_channels = [64, 128, 256]
        elif name_of_yolo_model == 'yolo11s':
            src_yolo_model = yolo_v11_s()
            feature_channels = [128, 256, 512]
        elif name_of_yolo_model == 'yolo11m':
            src_yolo_model = yolo_v11_m()
            feature_channels = [256, 512, 512]
        elif name_of_yolo_model == 'yolo11l':
            src_yolo_model = yolo_v11_l()
            feature_channels = [256, 512, 512]
        elif name_of_yolo_model == 'yolo11x':
            src_yolo_model = yolo_v11_x()
            feature_channels = [384, 768, 768]
        else:
            raise ValueError(f'Invalid backbone: {name_of_yolo_model}')
        src_yolo_model.cuda()
        # load weights
        print(f'Loading weights from {pretrained_weights_path}')
        load_weight(src_yolo_model, pretrained_weights_path)

        # copy the net and fpn
        self.net = copy.deepcopy(src_yolo_model.net)
        self.fpn = copy.deepcopy(src_yolo_model.fpn)
        if config_head['version'] == 'ver0':
            self.head = GuidewireDetectionHeadVer0(input_image_shape,
                                                   feature_channels,
                                                   config_head,
                                                   from_logits)
        elif config_head['version'] == 'ver1':
            self.head = GuidewireDetectionHeadVer1(input_image_shape,
                                                   feature_channels,
                                                   config_head,
                                                   from_logits)
        elif config_head['version'] == 'ver2':
            self.head = GuidewireDetectionHeadVer2(input_image_shape,
                                                   feature_channels,
                                                   config_head,
                                                   from_logits)
        elif config_head['version'] == 'ver3':
            self.head = GuidewireDetectionHeadVer3(input_image_shape,
                                                   feature_channels,
                                                   config_head,
                                                   from_logits)
        elif config_head['version'] == 'ver4':
            self.head = GuidewireDetectionHeadVer4(input_image_shape,
                                                   feature_channels,
                                                   config_head,
                                                   from_logits)
        else:
            raise ValueError(f'Invalid head version: {config_head["version"]}')
        # delete the original model
        del src_yolo_model

    def forward(self, x):
        # first of all, convert the grayscale image to RGB image
        x = x.repeat(1, 3, 1, 1)
        # pass it to the backbone
        y = self.net(x)
        y = self.fpn(y)
        # to custom head, we pass x (input image)
        # together with y (output features)
        z = [x] + list(y)
        return self.head(z)

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self

    def freeze_backbone(self):
        for param in self.net.parameters():
            param.requires_grad = False
        for param in self.fpn.parameters():
            param.requires_grad = False
        return self

    def unfreeze_backbone(self):
        for param in self.net.parameters():
            param.requires_grad = True
        for param in self.fpn.parameters():
            param.requires_grad = True
        return self