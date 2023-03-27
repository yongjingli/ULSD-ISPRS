from network.repvgg import get_RepVGG_func_by_name
from network.layers import *


class Lld_Repvgg_Large(nn.Module):
    def __init__(self, cfg=None):
        super(Lld_Repvgg_Large, self).__init__()

        # self.backbone = get_RepVGG_func_by_name('RepVGG-B1g2')(deploy=False)
        # in_channels = [64, 128, 256, 512, 2048]  # B1g2
        # pretrain_path = "/home/dev/data_disk/liyj/data/pretrain_model_data/RepVGG-B1g2-train.pth"
        if cfg is not None:
            self.backbone = get_RepVGG_func_by_name(cfg.model.sub_name)(deploy=False)
        else:
            self.backbone = get_RepVGG_func_by_name("RepVGG-A2")(deploy=False)

        in_channels = [64, 96, 192, 384, 1408]  # RepVGG-A2

        # if int(os.getenv('FuYao', 0)):
        #     pretrain_path = "/dataset/liyj/data/pretrain_models/checkpoints_line/RepVGG-A2-train.pth"
        # else:
        #     pretrain_path = "/userdata/liyj/data/pretrain_model_data/RepVGG-A2-train.pth"

        if cfg is not None:
            pretrain_path = cfg.model.pretrain_path
        else:
            pretrain_path = None

        # load pretrain model
        if pretrain_path is not None:
            checkpoint = torch.load(pretrain_path)
            self.backbone.load_state_dict(checkpoint, strict=True)

        ## A, B
        self.block15 = BlockTypeA(in_c1=in_channels[3], in_c2=in_channels[4],
                                  out_c1=in_channels[3], out_c2=in_channels[3])
        self.block16 = BlockTypeB(in_channels[3]*2, in_channels[3])

        ## A, B
        self.block17 = BlockTypeA(in_c1=in_channels[2],  in_c2=in_channels[3],
                                  out_c1=in_channels[2],  out_c2=in_channels[2])
        self.block18 = BlockTypeB(in_channels[2]*2, in_channels[2])

        ## A, B
        self.block19 = BlockTypeA(in_c1=in_channels[1],  in_c2=in_channels[2],
                                  out_c1=in_channels[1],  out_c2=in_channels[1])
        self.block20 = BlockTypeB(in_channels[1]*2, in_channels[1])

        # 5 + cls_num
        self.num_classes = cfg.model.num_classes
        self.block23 = BlockTypeC(in_channels[1], 5 + self.num_classes)
        # self.block23 = BlockTypeC(in_channels[1], 117)
        # self.block23 = BlockTypeC(in_channels[0], 41)

    def forward(self, x):
        # big: down ratio: 4, 8, 16, 32, channels: 192, 384, 768, 1536
        # base: [128, 256, 512, 1024]
        c1, c2, c3, c4, c5 = self.backbone.forward_stages(x)

        x = self.block15(c4, c5)
        x = self.block16(x)

        x = self.block17(c3, x)
        x = self.block18(x)

        x = self.block19(c2, x)
        x = self.block20(x)

        # x = self.block21(c1, x)
        # x = self.block22(x)

        x = self.block23(x)
        return x


if __name__ == "__main__":
    print("Start")
    lld_repvgg_model = Lld_Repvgg_Large()
    input_t = torch.ones((1, 3, 352, 640))
    output = lld_repvgg_model(input_t)

    print(output.shape)
    print("Done")