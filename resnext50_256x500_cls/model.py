#https://github.com/junfu1115/DANet

from common  import *
from dataset import *
from seresnext  import *

# overwrite ...
from dataset import null_collate as null_collate0
def null_collate(batch):
    input, truth_label, truth_mask, infor = null_collate0(batch)
    with torch.no_grad():
        arange = torch.FloatTensor([1,2,3,4]).to(truth_mask.device).view(1,4,1,1).long()
        m = truth_mask.repeat(1,4,1,1)
        m = (m==arange).float()
        truth_attention = F.avg_pool2d(m,kernel_size=(32,32),stride=(32,32))
        truth_attention = (truth_attention > 0/(32*32)).float()

        #relabel for augmentation cropping, etc
        truth_label = m.sum(dim=[2,3])
        truth_label = (truth_label > 1).float()

    return input, truth_label, truth_mask, truth_attention, infor


####################################################################################################


class Net(nn.Module):
    def load_pretrain(self, skip=['logit.'], is_print=True):
        load_pretrain(self, skip, pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=is_print)


    def __init__(self, num_class=4):
        super(Net, self).__init__()

        e = ResNext50()
        self.block0 = e.block0
        self.block1 = e.block1
        self.block2 = e.block2
        self.block3 = e.block3
        self.block4 = e.block4
        e = None  #dropped

        self.feature = nn.Conv2d(2048, 64, kernel_size=1) #dummy conv for dim reduction
        self.logit   = nn.Conv2d(64, num_class, kernel_size=1)



    def forward(self, x):
        batch_size,C,H,W = x.shape
        x = x.clone()
        x = x-torch.FloatTensor(IMAGE_RGB_MEAN).to(x.device).view(1,-1,1,1)
        x = x/torch.FloatTensor(IMAGE_RGB_STD).to(x.device).view(1,-1,1,1)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.dropout(x,0.5,training=self.training)
        x = F.avg_pool2d(x, kernel_size=(8, 13),stride=(8, 8))
        #x = F.adaptive_avg_pool2d(x, 1)
        x = self.feature(x)

        logit = self.logit(x) #.view(batch_size,-1)
        return logit



def criterion_label(logit, truth, weight=None):
    batch_size,num_class = logit.shape[:2]
    logit = logit.view(batch_size,num_class)
    truth = truth.view(batch_size,num_class)

    if weight is None: weight=[1,1,1,1]
    weight = torch.FloatTensor(weight).to(truth.device).view(1,-1)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')

    loss = loss*weight
    loss = loss.mean()
    return loss

#----
def metric_label(probability, truth, threshold=0.5):
    batch_size=len(truth)

    with torch.no_grad():
        probability = probability.view(batch_size,4)
        truth = truth.view(batch_size,4)

        #----
        neg_index = (truth==0).float()
        pos_index = 1-neg_index
        num_neg = neg_index.sum(0)
        num_pos = pos_index.sum(0)

        #----
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives
        tn = tn.sum(0)
        tp = tp.sum(0)

        #----
        tn = tn.data.cpu().numpy()
        tp = tp.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().astype(np.int32)
        num_pos = num_pos.data.cpu().numpy().astype(np.int32)

    return tn,tp, num_neg,num_pos



##############################################################################################
def make_dummy_data(batch_size=8):

    image_id = np.array([
        i +'.jpg' for i in [
            '0a8fddf7a', '0a29ef6f9', '0a46cc4bf', '0a058fcb6', '0a65bd8d4', '0a427a066', '0a6324223', '0b89f99d7',
            '00ac8372f', '1ae56dead', '1b7bec2ba', '1bdb7f26f', '1cac6e1f3', '1d34ad26c', '1d83b44be', '1e75373b2',
            '0b4c8e681', '0b5018316', '2b01fd731', '0cb590f8e', '0d4866e3c', '0e106d482', '0ebdc1277', '1bed9264f',
            '0a9aaba9a', '0a26aceb2', '0a405b396', '0aa7955fd', '0bda9a0eb', '0c2522533', '0cd22bad5', '0ce3a145f',
            '0adc17f1d', '0b56da4ff', '0be9bad7b', '0c888ecb5', '0d4eae8de', '0d78ac743', '0d51538b9', '0ddbc9fb5',
        ]
    ]).reshape(5,-1).T.reshape(-1).tolist()


    DATA_DIR = '/root/share/project/kaggle/2019/steel/data'
    folder = 'train_images'

    df = pd.read_csv(DATA_DIR+'/train.csv').fillna('')
    df = df_loc_by_list(df, 'ImageId_ClassId', [ i + '_%d'%c  for i in image_id for c in [1,2,3,4] ])
    df = df.reset_index(drop=True)
    #print(df)
    #exit(0)

    batch = []
    for b in range(0, batch_size):
        num_image = len(df)//4
        i = b%num_image

        image_id = df['ImageId_ClassId'].values[i*4][:-2]
        rle =  df['EncodedPixels'].values[i*4:(i+1)*4:]
        image = cv2.imread(DATA_DIR + '/%s/%s'%(folder,image_id), cv2.IMREAD_COLOR)
        label = [ 0 if r=='' else 1 for r in rle]
        mask  = np.array([run_length_decode(r, height=256, width=1600, fill_value=c) for c,r in zip([1,2,3,4],rle)])


        #---
        #crop to 256x400
        w=400
        mask_sum = mask.sum(1).sum(0)
        mask_sum = mask_sum.cumsum()
        mask_sum = mask_sum[w:]-mask_sum[:-w]
        x = np.argmax(mask_sum)
        image = image[:,x:x+w]
        mask = mask[:,:,x:x+w]

        zz=0
        #---
        mask  = mask.max(0, keepdims=0)
        infor = Struct(
            index    = i,
            folder   = folder,
            image_id = image_id,
        )
        batch.append([image,label,mask,infor])

    input, truth_label, truth_mask, truth_attention, infor = null_collate(batch)
    input = input.cuda()
    truth_label = truth_label.cuda()
    truth_mask  = truth_mask.cuda()
    truth_attention = truth_attention.cuda()

    return input, truth_label, truth_mask, truth_attention, infor



#########################################################################
def run_check_basenet():
    net = Net()
    print(net)
    #---
    if 1:
        print(net)
        print('')

        print('*** print key *** ')
        state_dict = net.state_dict()
        keys = list(state_dict.keys())
        #keys = sorted(keys)
        for k in keys:
            if any(s in k for s in [
                'num_batches_tracked'
                # '.kernel',
                # '.gamma',
                # '.beta',
                # '.running_mean',
                # '.running_var',
            ]):
                continue

            p = state_dict[k].data.cpu().numpy()
            print(' \'%s\',\t%s,'%(k,tuple(p.shape)))
        print('')

    net.load_pretrain(skip=['logit'])



def run_check_net():

    batch_size = 1
    C, H, W    = 3, 256, 1600
    num_class  = 4

    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = np.random.uniform(-1,1,(batch_size,C, H, W ))
    input = torch.from_numpy(input).float().cuda()

    net = Net(num_class=num_class).cuda()
    net.eval()

    with torch.no_grad():
        logit_label = net(input)

    print('')
    print('input: ',input.shape)
    print('logit: ',logit_label.shape)
    #print(net)


def run_check_train():


    loss_weight = [1,1,1,1]
    if 1:
        input, truth_label, truth_mask, truth_attention, infor = make_dummy_data(batch_size=30)
        batch_size, C, H, W  = input.shape

        print('input :', input.shape)
        print('truth_label :',truth_label.shape)
        print('(count)     :',truth_label.sum(0))
        print('truth_mask  :',truth_mask.shape)
        print('')
    #---

    net = Net().cuda()
    net.load_pretrain(is_print=False)#

    net = net.eval()
    with torch.no_grad():
        logit_label = net(input)
        loss = criterion_label(logit_label, truth_label)

        probability_label = torch.sigmoid(logit_label)
        tn,tp, num_neg,num_pos = metric_label(probability_label, truth_label)

        print('loss = %0.5f'%loss.item())
        print('tn,tp = [%0.3f,%0.3f,%0.3f,%0.3f], [%0.3f,%0.3f,%0.3f,%0.3f] '%(*(tn/(num_neg+1e-8)),*(tp/(num_pos+1e-8))))
        print('num_pos,num_neg = [%d,%d,%d,%d], [%d,%d,%d,%d] '%(*num_neg,*num_pos))
        print('')


    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=0.1, momentum=0.9, weight_decay=0.0001)

    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =',batch_size)
    print('----------------------------------------------------------------------')
    print('[iter ]  loss     |          [tn1,2,3,4  : tp1,2,3,4]       ')
    print('----------------------------------------------------------------------')
          #[00040]  0.63579  | [1.00,1.00,1.00,1.00 : 1.00,1.00,1.00,1.00]


    i=0
    optimizer.zero_grad()
    while i<=50:

        net.train()
        optimizer.zero_grad()

        logit_label = net(input)
        loss = criterion_label(logit_label, truth_label)

        probability_label = torch.sigmoid(logit_label)
        tn,tp, num_neg,num_pos = metric_label(probability_label, truth_label)

        (loss).backward()
        optimizer.step()

        if i%10==0:
            print('[%05d] %8.5f  | [%0.2f,%0.2f,%0.2f,%0.2f : %0.2f,%0.2f,%0.2f,%0.2f] '%(
                i,
                loss.item(),
                *(tn/(num_neg+1e-8)),*(tp/(num_pos+1e-8)),
            ))
        i = i+1
    print('')


    if 1:
        #net.eval()
        logit_label = net(input)
        probability_label = torch.sigmoid(logit_label)

        probability_label = probability_label.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        truth_mask  = truth_mask.data.cpu().numpy()

        image = input_to_image(input)
        for b in range(batch_size):
            print('%d ------ '%(b))
            result = draw_predict_result_label(image[b], truth_label[b], truth_mask[b], probability_label[b])
            image_show('result',result, resize=1)
            cv2.waitKey(0)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_basenet()
    #run_check_net()
    run_check_train()


    print('\nsucess!')


