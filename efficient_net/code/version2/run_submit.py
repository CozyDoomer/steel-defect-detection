import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from common  import *
from dataset import *
from model   import *

from etc import *
from defect import *


######################################################################################
TEMPERATE=0


######################################################################################


def do_evaluate_segmentation(net, test_dataset, augment=[], out_dir=None):

    test_loader = DataLoader(
        test_dataset,
        sampler     = SequentialSampler(test_dataset),
        batch_size  = 4,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,
        collate_fn  = null_collate
    )
    #----

    #def sharpen(p,t=0):
    def sharpen(p,t=TEMPERATE):
        if t!=0:
            return p**t
        else:
            return p


    test_num  = 0
    test_id   = []
    test_probability_label = [] # 8bit
    test_probability_mask  = [] # 8bit
    test_truth_label = [] # 8bit
    test_truth_mask  = [] # 8bit

    start_timer = timer()
    for t, (input, truth_label, truth_mask, truth_attention, infor) in enumerate(test_loader):

        batch_size,C,H,W = input.shape
        input = input.cuda()

        with torch.no_grad():
            net.eval()

            num_augment = 0
            if 1: #  null
                logit =  data_parallel(net,input)  #net(input)
                probability = torch.softmax(logit,1)

                probability_mask  = sharpen(probability,0)
                probability_label = probability_mask_to_probability_label(probability)
                num_augment+=1

            if 'flip_lr' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[3]))
                probability  = torch.softmax(torch.flip(logit,dims=[3]),1)

                probability_mask  += sharpen(probability)
                probability_label += probability_mask_to_probability_label(probability)
                num_augment+=1

            if 'flip_ud' in augment:
                logit = data_parallel(net,torch.flip(input,dims=[2]))
                probability = torch.softmax(torch.flip(logit,dims=[2]),1)

                probability_mask += sharpen(probability)
                probability_label+= probability_mask_to_probability_label(probability)
                num_augment+=1

            #---
            probability_mask  = probability_mask/num_augment
            probability_label = probability_label/num_augment

        #---
        batch_size  = len(infor)
        truth_label = truth_label.data.cpu().numpy().astype(np.uint8)
        truth_mask  = truth_mask.data.cpu().numpy().astype(np.uint8)
        probability_mask = (probability_mask.data.cpu().numpy()*255).astype(np.uint8)
        probability_label = (probability_label.data.cpu().numpy()*255).astype(np.uint8)

        test_id.extend([i.image_id for i in infor])
        test_truth_label.append(truth_label)
        test_truth_mask.append(truth_mask)
        test_probability_label.append(probability_label)
        test_probability_mask.append(probability_mask)
        test_num += batch_size

        #---
        print('\r %4d / %4d  %s'%(
             test_num, len(test_loader.dataset), time_to_str((timer() - start_timer),'min')
        ),end='',flush=True)

    assert(test_num == len(test_loader.dataset))
    print('')

    start_timer = timer()
    test_truth_label = np.concatenate(test_truth_label)
    test_truth_mask  = np.concatenate(test_truth_mask)
    test_probability_label = np.concatenate(test_probability_label)
    test_probability_mask = np.concatenate(test_probability_mask)
    print(time_to_str((timer() - start_timer),'sec'))

    return test_id, test_truth_label, test_truth_mask, test_probability_label, test_probability_mask


######################################################################################
def run_submit_segmentation(

):
    train_split = ['valid_b0_1000.npy',]

    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h/checkpoint/00051000_model.pth'

    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb1'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb1/checkpoint/00053000_model.pth'

    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb2a'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb2a/checkpoint/00080000_model.pth'
         #'/root/share/project/kaggle/2019/steel/result15/efficientb5-fpn-crop256x400-foldb2/checkpoint/00065000_model.pth'
         #'/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb1/submit/test',
         #'/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb3/submit/test',

    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-semi-crop256x400-foldb2'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-semi-crop256x400-foldb2/checkpoint/00032000_model.pth'
    #------

    # initial_checkpoint = \
    #      '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb-x2/checkpoint/00032000_model.pth'
         #'/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb-x2/checkpoint/00007000_model.pth'
         #'/root/share/project/kaggle/2019/steel/result15/efficientb5-fpn-crop256x400-foldb2/checkpoint/00065000_model.pth'




    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb2'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb2/old_checkpoint/00065000_model.pth'
    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h/checkpoint/00051000_model.pth'
    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb1'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb1/checkpoint/00053000_model.pth'

    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb3'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb3/checkpoint/00067000_model.pth'
    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb-x2a'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb-x2a/checkpoint/00041000_model.pth'


    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0-gather'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h/checkpoint/00051000_model.pth'



    #----
    # out_dir = \
    #      '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h'
    # initial_checkpoint = \
    #      '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0h/checkpoint/00051000_model.pth'


    out_dir = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0-mish'
    initial_checkpoint = \
         '/root/share/project/kaggle/2019/steel/result99/efficientb5-fpn-crop256x400-foldb0-mish/checkpoint/00056000_model.pth'



    ###############################################################3

    augment = ['null', 'flip_lr','flip_ud']  #['null'] #
    mode = 'test' #'train' # 'test'
    mode_folder = 'test-tta' #tta


    #---

    ## setup
    os.makedirs(out_dir +'/submit/%s'%(mode_folder), exist_ok=True)

    log = Logger()
    log.open(out_dir+'/log.submit.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % __file__)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## dataset -------

    log.write('** dataset setting **\n')
    if mode == 'train':
        test_dataset = SteelDataset(
            mode    = 'train',
            csv     = ['train.csv',],
            split   = train_split,
            augment = None,
        )

    if mode == 'test':
        test_dataset = SteelDataset(
            mode    = 'test',
            csv     = ['sample_submission.csv',],
            split   = ['test_1801.npy',],
            augment = None, #
        )

    log.write('test_dataset : \n%s\n'%(test_dataset))
    log.write('\n')
    #exit(0)


    ## start testing here! ##############################################
    #

    #---
    threshold_label      = [ 0.75, 0.99, 0.50, 0.50,]
    threshold_mask_pixel = [ 0.50, 0.50, 0.50, 0.50,]
    threshold_mask_size  = [    1,    1,    1,    1,]

    threshold_mask_pixel = [ 0.40, 0.40, 0.40, 0.40,]

    image_id = read_list_from_file(out_dir + '/submit/%s/image_id.txt'%(mode_folder))
    probability_label = np.load(out_dir + '/submit/%s/probability_label.uint8.npz'%(mode_folder))['arr_0']
    probability_mask  = np.load(out_dir + '/submit/%s/probability_mask.uint8.npz'%(mode_folder))['arr_0']
    if mode == 'train':
        truth_label       = np.load(out_dir + '/submit/%s/truth_label.uint8.npz'%(mode_folder))['arr_0']
        truth_mask        = np.load(out_dir + '/submit/%s/truth_mask.uint8.npz'%(mode_folder))['arr_0']

    num_test= len(image_id)

    # value = np.max(probability_mask,1,keepdims=True)
    # value = probability_mask*(value==probability_mask)
    probability_mask = probability_mask[:,1:] #remove background class

    if mode == 'train':
        index = np.ones((num_test,4,256,1600),np.uint8)*np.array([1,2,3,4],np.uint8).reshape(1,4,1,1)
        truth_mask = truth_mask==index

    # inspect here !!!  ###################
    print('')
    log.write('submitting .... @ %s\n'%str(augment))
    log.write('threshold_label = %s\n'%str(threshold_label))
    log.write('threshold_mask_pixel = %s\n'%str(threshold_mask_pixel))
    log.write('threshold_mask_size  = %s\n'%str(threshold_mask_size))
    log.write('\n')

    if mode == 'train':

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)

        log.write('** threshold_label **\n')
        kaggle, result = compute_metric_label(truth_label, predict_label)
        text = summarise_metric_label(kaggle, result)
        log.write('\n%s'%(text))

        auc, result = compute_roc_label(truth_label, probability_label)
        text = summarise_roc_label(auc, result)
        log.write('\n%s'%(text))

        log.write('** threshold_pixel **\n')
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s'%(text))

        #-----

        log.write('** threshold_pixel + threshold_label **\n')
        predict_mask = predict_mask * predict_label.reshape(-1,4,1,1)
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s'%(text))

        #-----

        log.write('** threshold_pixel + threshold_label + threshold_size **\n')
        predict_mask = remove_small(predict_mask, threshold_mask_size)
        kaggle, result = compute_metric_mask(truth_mask, predict_mask)
        text = summarise_metric_mask(kaggle, result)
        log.write('\n%s'%(text))


    ###################

    if mode =='test':
        log.write('test submission .... @ %s\n'%str(augment))
        csv_file = out_dir +'/submit/%s/efficientnetb5-fpn35.csv'%(mode_folder)

        predict_label = probability_label>(np.array(threshold_label)*255).astype(np.uint8).reshape(1,4)
        predict_mask  = probability_mask>(np.array(threshold_mask_pixel)*255).astype(np.uint8).reshape(1,4,1,1)

        image_id_class_id = []
        encoded_pixel = []
        for b in range(len(image_id)):
            for c in range(4):
                image_id_class_id.append(image_id[b]+'_%d'%(c+1))

                if predict_label[b,c]==0:
                    rle=''
                else:
                    rle = run_length_encode(predict_mask[b,c])
                encoded_pixel.append(rle)

        df = pd.DataFrame(zip(image_id_class_id, encoded_pixel), columns=['ImageId_ClassId', 'EncodedPixels'])
        df.to_csv(csv_file, index=False)

        ## print statistics ----
        print('initial_checkpoint=%s'%initial_checkpoint)
        text = summarise_submission_csv(df)
        log.write('\n')
        log.write('%s'%(text))

        ##evalue based on probing results
        text = do_local_submit(image_id, predict_label,predict_mask)
        log.write('\n')
        log.write('%s'%(text))

    exit(0)
 




# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_submit_segmentation()
  
