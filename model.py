
from layer import *


class SCFEGAN(nn.Module):
    def __init__(self, in_channels=4, nker=64, out_channels=3):
        super(SCFEGAN, self).__init__()

        #encoder
        self.enc1 = GatedConv2d(in_channels=in_channels, out_channels=nker, kernel_size=7, stride=2, padding=3)
        # feature=256 channels=64
        self.enc2 = GatedConv2d(in_channels=nker, out_channels=2 * nker, kernel_size=5, stride=2, padding=2)
        # 128 128
        self.enc3 = GatedConv2d(in_channels=2*nker, out_channels=4*nker, kernel_size=5, stride=2, padding=2)
        # 64 256
        self.enc4 = GatedConv2d(in_channels=4*nker, out_channels=8*nker, kernel_size=3, stride=2, padding=1)
        # 32 512
        self.enc5 = GatedConv2d(in_channels=8*nker, out_channels=8*nker, kernel_size=3, stride=2, padding=1)
        # 16 512
        self.enc6 = GatedConv2d(in_channels=8*nker, out_channels=8*nker, kernel_size=3, stride=2, padding=1)
        # 8 512
        self.enc7 = GatedConv2d(in_channels=8*nker, out_channels=8*nker, kernel_size=3, stride=2, padding=1) # 4 512


        #decoder

        self.dec7_1 = DeGatedConv(in_channels=8*nker, out_channels= 8*nker, kernel_size=2, stride=2, padding=0)
        self.dec7_2 = GatedConv2d(in_channels= 16*nker, out_channels=8*nker, kernel_size=3, stride=1, padding=1)

        self.dec6_1 = DeGatedConv(in_channels=8*nker, out_channels=8*nker, kernel_size=2, stride=2, padding=0)
        self.dec6_2 = GatedConv2d(in_channels=16*nker, out_channels= 8*nker, kernel_size=3, stride=1, padding=1)

        self.dec5_1 = DeGatedConv(in_channels=8*nker, out_channels=8*nker, kernel_size=2, stride=2, padding=0)
        self.dec5_2 = GatedConv2d(in_channels=16*nker, out_channels=8*nker, kernel_size=3, stride=1, padding=1)

        self.dec4_1 = DeGatedConv(in_channels=8*nker, out_channels=4*nker, kernel_size=2, stride=2, padding=0)
        self.dec4_2 = GatedConv2d(in_channels=8*nker, out_channels=4*nker, kernel_size=3, stride=1, padding=1)

        self.dec3_1 = DeGatedConv(in_channels=4*nker, out_channels=2*nker, kernel_size=2, stride=2, padding=0)
        self.dec3_2 = GatedConv2d(in_channels=4*nker, out_channels=2*nker, kernel_size=3, stride=1, padding=1)

        self.dec2_1 = DeGatedConv(in_channels=2*nker, out_channels=nker, kernel_size=2, stride=2, padding=0)
        # self.dec2_2 = GatedConv2d(in_channels=2*nker, out_channels=nker, kernel_size=3, stride=1, padding=1)

        self.dec1_1 = DeGatedConv(in_channels=2*nker, out_channels=in_channels, kernel_size=2, stride=2, padding=0)
        self.dec1_2 = GatedConv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  activation=None)

        self.tanh = nn.Tanh()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)


        dec7_1 = self.dec7_1(enc7)
        cat7 = torch.cat((dec7_1, enc6), dim=1)
        dec7_2 = self.dec7_2(cat7)


        dec6_1 = self.dec6_1(dec7_2)
        cat6 = torch.cat((dec6_1, enc5), dim=1)
        dec6_2 = self.dec6_2(cat6)

        dec5_1 = self.dec5_1(dec6_2)
        cat5 = torch.cat((dec5_1, enc4), dim=1)
        dec5_2 = self.dec5_2(cat5)

        dec4_1 = self.dec4_1(dec5_2)
        cat4 = torch.cat((dec4_1, enc3), dim=1)
        dec4_2 = self.dec4_2(cat4)


        dec3_1 = self.dec3_1(dec4_2)
        cat3 = torch.cat((dec3_1, enc2), dim=1)
        dec3_2 = self.dec3_2(cat3)


        dec2_1 = self.dec2_1(dec3_2)
        cat2 = torch.cat((dec2_1, enc1), dim=1)
        # dec2_2 = self.dec2_2(cat2)

        dec1_1 = self.dec1_1(cat2)
        cat1 = torch.cat((dec1_1, x), dim=1)
        dec1_2 = self.dec1_2(cat1)

        output = self.tanh(dec1_2)


        # print('enc1 shape = {}'.format(enc1.shape))
        # print('enc2 shape = {}'.format(enc2.shape))
        # print('enc3 shape = {}'.format(enc3.shape))
        # print('{} shape = {}'.format('enc4', enc4.shape))
        # print('{} shape = {}'.format('enc5', enc5.shape))
        # print('{} shape = {}'.format('enc6', enc6.shape))
        # print('{} shape = {}'.format('enc7', enc7.shape))
        #
        # print('{} shape = {}'.format('dec7_1', dec7_1.shape))
        # print('{} shape = {}'.format('dec7_2', dec7_2.shape))
        #
        # print('{} shape = {}'.format('dec6_1', dec6_1.shape))
        # print('{} shape = {}'.format('dec6_2', dec6_2.shape))
        #
        # print('{} shape = {}'.format('dec5_1', dec5_1.shape))
        # print('{} shape = {}'.format('dec5_2', dec5_2.shape))
        #
        # print('{} shape = {}'.format('dec4_1', dec4_1.shape))
        # print('{} shape = {}'.format('dec4_2', dec4_2.shape))
        #
        # print('{} shape = {}'.format('dec3_1', dec3_1.shape))
        # print('{} shape = {}'.format('dec3_2', dec3_2.shape))
        #
        # print('{} shape = {}'.format('dec2_1', dec2_1.shape))
        #
        # print('{} shape = {}'.format('dec1_1', dec1_1.shape))
        # print('{} shape = {}'.format('dec1_2', dec1_2.shape))
        #
        # print('{} shape = {}'.format('tanh', output.shape))


        return output


class SNDiscrim(nn.Module):
    def __init__(self, in_channels=4, nkr=64):
        super(SNDiscrim, self).__init__()

        self.snconv1 = SNConv(in_channels=in_channels, out_channels=nkr, kernel_size=3, padding=1, stride=1)
        self.snconv2 = SNConv(in_channels=nkr, out_channels=2*nkr, kernel_size=3, padding=1, stride=2)
        self.snconv3 = SNConv(in_channels=2*nkr, out_channels=4*nkr, kernel_size=3, padding=1, stride=2)
        self.snconv4 = SNConv(in_channels=4*nkr, out_channels=4*nkr, kernel_size=3, padding=1, stride=2)
        self.snconv5 = SNConv(in_channels=4*nkr, out_channels=4*nkr, kernel_size=3, padding=1, stride=2)
        self.snconv6 = SNConv(in_channels=4*nkr, out_channels=4*nkr, kernel_size=3, padding=1, stride=2)

    def forward(self, data):
        x = self.snconv1(data)
        x = self.snconv2(x)
        x = self.snconv3(x)
        x = self.snconv4(x)
        x = self.snconv5(x)
        x = self.snconv6(x)

        x = torch.sigmoid(x)

        return(x)


class SCFEGAN256(nn.Module):
    def __init__(self, in_channels=4, nker=64, out_channels=3):
        super(SCFEGAN256, self).__init__()

        self.enc1 = GatedConv2d(in_channels=in_channels, out_channels=nker, kernel_size=7, stride=2, padding=3)
        # feature=128 channels=64
        self.enc2 = GatedConv2d(in_channels=nker, out_channels=2 * nker, kernel_size=5, stride=2, padding=2)
        # 64
        self.enc3 = GatedConv2d(in_channels=2 * nker, out_channels=4 * nker, kernel_size=5, stride=2, padding=2)
        # 32 256
        self.enc4 = GatedConv2d(in_channels=4 * nker, out_channels=8 * nker, kernel_size=3, stride=2, padding=1)
        # 16 512
        self.enc5 = GatedConv2d(in_channels=8 * nker, out_channels=8 * nker, kernel_size=3, stride=2, padding=1)
        # 8 512
        self.enc6 = GatedConv2d(in_channels=8 * nker, out_channels=8 * nker, kernel_size=3, stride=2, padding=1)
        # 4

        self.dec7_1 = DeGatedConv(in_channels=8 * nker, out_channels=8 * nker, kernel_size=2, stride=2, padding=0)
        self.dec7_2 = GatedConv2d(in_channels=16 * nker, out_channels=8 * nker, kernel_size=3, stride=1, padding=1)

        self.dec6_1 = DeGatedConv(in_channels=8*nker, out_channels=8*nker, kernel_size=2, stride=2, padding=0)
        self.dec6_2 = GatedConv2d(in_channels=16*nker, out_channels= 8*nker, kernel_size=3, stride=1, padding=1)

        self.dec5_1 = DeGatedConv(in_channels=8*nker, out_channels=8*nker, kernel_size=2, stride=2, padding=0)
        self.dec5_2 = GatedConv2d(in_channels=16*nker, out_channels=8*nker, kernel_size=3, stride=1, padding=1)

        self.dec4_1 = DeGatedConv(in_channels=8*nker, out_channels=4*nker, kernel_size=2, stride=2, padding=0)
        self.dec4_2 = GatedConv2d(in_channels=8*nker, out_channels=4*nker, kernel_size=3, stride=1, padding=1)

        self.dec3_1 = DeGatedConv(in_channels=4*nker, out_channels=2*nker, kernel_size=2, stride=2, padding=0)
        self.dec3_2 = GatedConv2d(in_channels=4*nker, out_channels=2*nker, kernel_size=3, stride=1, padding=1)

        self.dec2_1 = DeGatedConv(in_channels=2*nker, out_channels=nker, kernel_size=2, stride=2, padding=0)
        # self.dec2_2 = GatedConv2d(in_channels=2*nker, out_channels=nker, kernel_size=3, stride=1, padding=1)

        self.dec1_1 = DeGatedConv(in_channels=2*nker, out_channels=in_channels, kernel_size=2, stride=2, padding=0)
        self.dec1_2 = GatedConv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                                  activation=None)

        self.tanh = nn.Tanh()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)

        dec6_1 = self.dec6_1(enc6)
        cat6 = torch.cat((dec6_1, enc5), dim=1)
        dec6_2 = self.dec6_2(cat6)

        dec5_1 = self.dec5_1(dec6_2)
        cat5 = torch.cat((dec5_1, enc4), dim=1)
        dec5_2 = self.dec5_2(cat5)

        dec4_1 = self.dec4_1(dec5_2)
        cat4 = torch.cat((dec4_1, enc3), dim=1)
        dec4_2 = self.dec4_2(cat4)

        dec3_1 = self.dec3_1(dec4_2)
        cat3 = torch.cat((dec3_1, enc2), dim=1)
        dec3_2 = self.dec3_2(cat3)

        dec2_1 = self.dec2_1(dec3_2)
        cat2 = torch.cat((dec2_1, enc1), dim=1)
        # dec2_2 = self.dec2_2(cat2)

        dec1_1 = self.dec1_1(cat2)
        cat1 = torch.cat((dec1_1, x), dim=1)
        dec1_2 = self.dec1_2(cat1)

        output = self.tanh(dec1_2)

        # print('enc1 shape = {}'.format(enc1.shape))
        # print('enc2 shape = {}'.format(enc2.shape))
        # print('enc3 shape = {}'.format(enc3.shape))
        # print('{} shape = {}'.format('enc4', enc4.shape))
        # print('{} shape = {}'.format('enc5', enc5.shape))
        # print('{} shape = {}'.format('enc6', enc6.shape))
        #
        # print('{} shape = {}'.format('dec6_1', dec6_1.shape))
        # print('{} shape = {}'.format('dec6_2', dec6_2.shape))
        #
        # print('{} shape = {}'.format('dec5_1', dec5_1.shape))
        # print('{} shape = {}'.format('dec5_2', dec5_2.shape))
        #
        # print('{} shape = {}'.format('dec4_1', dec4_1.shape))
        # print('{} shape = {}'.format('dec4_2', dec4_2.shape))
        #
        # print('{} shape = {}'.format('dec3_1', dec3_1.shape))
        # print('{} shape = {}'.format('dec3_2', dec3_2.shape))
        #
        # print('{} shape = {}'.format('dec2_1', dec2_1.shape))
        #
        # print('{} shape = {}'.format('dec1_1', dec1_1.shape))
        # print('{} shape = {}'.format('dec1_2', dec1_2.shape))
        #
        # print('{} shape = {}'.format('tanh', output.shape))



        return output


