from common import *
from lib.net.sync_bn.nn import BatchNorm2dSync as SynchronizedBatchNorm2d


###############################################################################

BatchNorm2d = SynchronizedBatchNorm2d
#BatchNorm2d = nn.BatchNorm2d

###############################################################################

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

IS_GATHER_EXCITE = False  # True  # False

###############################################################################
CONVERSION=[
 'block0.0.weight',	(64, 3, 7, 7),	 'layer0.conv1.weight',	(64, 3, 7, 7),
 'block0.1.weight',	(64,),	 'layer0.bn1.weight',	(64,),
 'block0.1.bias',	(64,),	 'layer0.bn1.bias',	(64,),
 'block0.1.running_mean',	(64,),	 'layer0.bn1.running_mean',	(64,),
 'block0.1.running_var',	(64,),	 'layer0.bn1.running_var',	(64,),
 'block1.1.conv_bn1.conv.weight',	(128, 64, 1, 1),	 'layer1.0.conv1.weight',	(128, 64, 1, 1),
 'block1.1.conv_bn1.bn.weight',	(128,),	 'layer1.0.bn1.weight',	(128,),
 'block1.1.conv_bn1.bn.bias',	(128,),	 'layer1.0.bn1.bias',	(128,),
 'block1.1.conv_bn1.bn.running_mean',	(128,),	 'layer1.0.bn1.running_mean',	(128,),
 'block1.1.conv_bn1.bn.running_var',	(128,),	 'layer1.0.bn1.running_var',	(128,),
 'block1.1.conv_bn2.conv.weight',	(128, 4, 3, 3),	 'layer1.0.conv2.weight',	(128, 4, 3, 3),
 'block1.1.conv_bn2.bn.weight',	(128,),	 'layer1.0.bn2.weight',	(128,),
 'block1.1.conv_bn2.bn.bias',	(128,),	 'layer1.0.bn2.bias',	(128,),
 'block1.1.conv_bn2.bn.running_mean',	(128,),	 'layer1.0.bn2.running_mean',	(128,),
 'block1.1.conv_bn2.bn.running_var',	(128,),	 'layer1.0.bn2.running_var',	(128,),
 'block1.1.conv_bn3.conv.weight',	(256, 128, 1, 1),	 'layer1.0.conv3.weight',	(256, 128, 1, 1),
 'block1.1.conv_bn3.bn.weight',	(256,),	 'layer1.0.bn3.weight',	(256,),
 'block1.1.conv_bn3.bn.bias',	(256,),	 'layer1.0.bn3.bias',	(256,),
 'block1.1.conv_bn3.bn.running_mean',	(256,),	 'layer1.0.bn3.running_mean',	(256,),
 'block1.1.conv_bn3.bn.running_var',	(256,),	 'layer1.0.bn3.running_var',	(256,),
 'block1.1.scale.fc1.weight',	(16, 256, 1, 1),	 'layer1.0.se_module.fc1.weight',	(16, 256, 1, 1),
 'block1.1.scale.fc1.bias',	(16,),	 'layer1.0.se_module.fc1.bias',	(16,),
 'block1.1.scale.fc2.weight',	(256, 16, 1, 1),	 'layer1.0.se_module.fc2.weight',	(256, 16, 1, 1),
 'block1.1.scale.fc2.bias',	(256,),	 'layer1.0.se_module.fc2.bias',	(256,),
 'block1.1.shortcut.conv.weight',	(256, 64, 1, 1),	 'layer1.0.downsample.0.weight',	(256, 64, 1, 1),
 'block1.1.shortcut.bn.weight',	(256,),	 'layer1.0.downsample.1.weight',	(256,),
 'block1.1.shortcut.bn.bias',	(256,),	 'layer1.0.downsample.1.bias',	(256,),
 'block1.1.shortcut.bn.running_mean',	(256,),	 'layer1.0.downsample.1.running_mean',	(256,),
 'block1.1.shortcut.bn.running_var',	(256,),	 'layer1.0.downsample.1.running_var',	(256,),
 'block1.2.conv_bn1.conv.weight',	(128, 256, 1, 1),	 'layer1.1.conv1.weight',	(128, 256, 1, 1),
 'block1.2.conv_bn1.bn.weight',	(128,),	 'layer1.1.bn1.weight',	(128,),
 'block1.2.conv_bn1.bn.bias',	(128,),	 'layer1.1.bn1.bias',	(128,),
 'block1.2.conv_bn1.bn.running_mean',	(128,),	 'layer1.1.bn1.running_mean',	(128,),
 'block1.2.conv_bn1.bn.running_var',	(128,),	 'layer1.1.bn1.running_var',	(128,),
 'block1.2.conv_bn2.conv.weight',	(128, 4, 3, 3),	 'layer1.1.conv2.weight',	(128, 4, 3, 3),
 'block1.2.conv_bn2.bn.weight',	(128,),	 'layer1.1.bn2.weight',	(128,),
 'block1.2.conv_bn2.bn.bias',	(128,),	 'layer1.1.bn2.bias',	(128,),
 'block1.2.conv_bn2.bn.running_mean',	(128,),	 'layer1.1.bn2.running_mean',	(128,),
 'block1.2.conv_bn2.bn.running_var',	(128,),	 'layer1.1.bn2.running_var',	(128,),
 'block1.2.conv_bn3.conv.weight',	(256, 128, 1, 1),	 'layer1.1.conv3.weight',	(256, 128, 1, 1),
 'block1.2.conv_bn3.bn.weight',	(256,),	 'layer1.1.bn3.weight',	(256,),
 'block1.2.conv_bn3.bn.bias',	(256,),	 'layer1.1.bn3.bias',	(256,),
 'block1.2.conv_bn3.bn.running_mean',	(256,),	 'layer1.1.bn3.running_mean',	(256,),
 'block1.2.conv_bn3.bn.running_var',	(256,),	 'layer1.1.bn3.running_var',	(256,),
 'block1.2.scale.fc1.weight',	(16, 256, 1, 1),	 'layer1.1.se_module.fc1.weight',	(16, 256, 1, 1),
 'block1.2.scale.fc1.bias',	(16,),	 'layer1.1.se_module.fc1.bias',	(16,),
 'block1.2.scale.fc2.weight',	(256, 16, 1, 1),	 'layer1.1.se_module.fc2.weight',	(256, 16, 1, 1),
 'block1.2.scale.fc2.bias',	(256,),	 'layer1.1.se_module.fc2.bias',	(256,),
 'block1.3.conv_bn1.conv.weight',	(128, 256, 1, 1),	 'layer1.2.conv1.weight',	(128, 256, 1, 1),
 'block1.3.conv_bn1.bn.weight',	(128,),	 'layer1.2.bn1.weight',	(128,),
 'block1.3.conv_bn1.bn.bias',	(128,),	 'layer1.2.bn1.bias',	(128,),
 'block1.3.conv_bn1.bn.running_mean',	(128,),	 'layer1.2.bn1.running_mean',	(128,),
 'block1.3.conv_bn1.bn.running_var',	(128,),	 'layer1.2.bn1.running_var',	(128,),
 'block1.3.conv_bn2.conv.weight',	(128, 4, 3, 3),	 'layer1.2.conv2.weight',	(128, 4, 3, 3),
 'block1.3.conv_bn2.bn.weight',	(128,),	 'layer1.2.bn2.weight',	(128,),
 'block1.3.conv_bn2.bn.bias',	(128,),	 'layer1.2.bn2.bias',	(128,),
 'block1.3.conv_bn2.bn.running_mean',	(128,),	 'layer1.2.bn2.running_mean',	(128,),
 'block1.3.conv_bn2.bn.running_var',	(128,),	 'layer1.2.bn2.running_var',	(128,),
 'block1.3.conv_bn3.conv.weight',	(256, 128, 1, 1),	 'layer1.2.conv3.weight',	(256, 128, 1, 1),
 'block1.3.conv_bn3.bn.weight',	(256,),	 'layer1.2.bn3.weight',	(256,),
 'block1.3.conv_bn3.bn.bias',	(256,),	 'layer1.2.bn3.bias',	(256,),
 'block1.3.conv_bn3.bn.running_mean',	(256,),	 'layer1.2.bn3.running_mean',	(256,),
 'block1.3.conv_bn3.bn.running_var',	(256,),	 'layer1.2.bn3.running_var',	(256,),
 'block1.3.scale.fc1.weight',	(16, 256, 1, 1),	 'layer1.2.se_module.fc1.weight',	(16, 256, 1, 1),
 'block1.3.scale.fc1.bias',	(16,),	 'layer1.2.se_module.fc1.bias',	(16,),
 'block1.3.scale.fc2.weight',	(256, 16, 1, 1),	 'layer1.2.se_module.fc2.weight',	(256, 16, 1, 1),
 'block1.3.scale.fc2.bias',	(256,),	 'layer1.2.se_module.fc2.bias',	(256,),
 'block2.0.conv_bn1.conv.weight',	(256, 256, 1, 1),	 'layer2.0.conv1.weight',	(256, 256, 1, 1),
 'block2.0.conv_bn1.bn.weight',	(256,),	 'layer2.0.bn1.weight',	(256,),
 'block2.0.conv_bn1.bn.bias',	(256,),	 'layer2.0.bn1.bias',	(256,),
 'block2.0.conv_bn1.bn.running_mean',	(256,),	 'layer2.0.bn1.running_mean',	(256,),
 'block2.0.conv_bn1.bn.running_var',	(256,),	 'layer2.0.bn1.running_var',	(256,),
 'block2.0.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.0.conv2.weight',	(256, 8, 3, 3),
 'block2.0.conv_bn2.bn.weight',	(256,),	 'layer2.0.bn2.weight',	(256,),
 'block2.0.conv_bn2.bn.bias',	(256,),	 'layer2.0.bn2.bias',	(256,),
 'block2.0.conv_bn2.bn.running_mean',	(256,),	 'layer2.0.bn2.running_mean',	(256,),
 'block2.0.conv_bn2.bn.running_var',	(256,),	 'layer2.0.bn2.running_var',	(256,),
 'block2.0.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.0.conv3.weight',	(512, 256, 1, 1),
 'block2.0.conv_bn3.bn.weight',	(512,),	 'layer2.0.bn3.weight',	(512,),
 'block2.0.conv_bn3.bn.bias',	(512,),	 'layer2.0.bn3.bias',	(512,),
 'block2.0.conv_bn3.bn.running_mean',	(512,),	 'layer2.0.bn3.running_mean',	(512,),
 'block2.0.conv_bn3.bn.running_var',	(512,),	 'layer2.0.bn3.running_var',	(512,),
 'block2.0.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.0.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.0.scale.fc1.bias',	(32,),	 'layer2.0.se_module.fc1.bias',	(32,),
 'block2.0.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.0.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.0.scale.fc2.bias',	(512,),	 'layer2.0.se_module.fc2.bias',	(512,),
 'block2.0.shortcut.conv.weight',	(512, 256, 1, 1),	 'layer2.0.downsample.0.weight',	(512, 256, 1, 1),
 'block2.0.shortcut.bn.weight',	(512,),	 'layer2.0.downsample.1.weight',	(512,),
 'block2.0.shortcut.bn.bias',	(512,),	 'layer2.0.downsample.1.bias',	(512,),
 'block2.0.shortcut.bn.running_mean',	(512,),	 'layer2.0.downsample.1.running_mean',	(512,),
 'block2.0.shortcut.bn.running_var',	(512,),	 'layer2.0.downsample.1.running_var',	(512,),
 'block2.1.conv_bn1.conv.weight',	(256, 512, 1, 1),	 'layer2.1.conv1.weight',	(256, 512, 1, 1),
 'block2.1.conv_bn1.bn.weight',	(256,),	 'layer2.1.bn1.weight',	(256,),
 'block2.1.conv_bn1.bn.bias',	(256,),	 'layer2.1.bn1.bias',	(256,),
 'block2.1.conv_bn1.bn.running_mean',	(256,),	 'layer2.1.bn1.running_mean',	(256,),
 'block2.1.conv_bn1.bn.running_var',	(256,),	 'layer2.1.bn1.running_var',	(256,),
 'block2.1.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.1.conv2.weight',	(256, 8, 3, 3),
 'block2.1.conv_bn2.bn.weight',	(256,),	 'layer2.1.bn2.weight',	(256,),
 'block2.1.conv_bn2.bn.bias',	(256,),	 'layer2.1.bn2.bias',	(256,),
 'block2.1.conv_bn2.bn.running_mean',	(256,),	 'layer2.1.bn2.running_mean',	(256,),
 'block2.1.conv_bn2.bn.running_var',	(256,),	 'layer2.1.bn2.running_var',	(256,),
 'block2.1.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.1.conv3.weight',	(512, 256, 1, 1),
 'block2.1.conv_bn3.bn.weight',	(512,),	 'layer2.1.bn3.weight',	(512,),
 'block2.1.conv_bn3.bn.bias',	(512,),	 'layer2.1.bn3.bias',	(512,),
 'block2.1.conv_bn3.bn.running_mean',	(512,),	 'layer2.1.bn3.running_mean',	(512,),
 'block2.1.conv_bn3.bn.running_var',	(512,),	 'layer2.1.bn3.running_var',	(512,),
 'block2.1.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.1.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.1.scale.fc1.bias',	(32,),	 'layer2.1.se_module.fc1.bias',	(32,),
 'block2.1.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.1.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.1.scale.fc2.bias',	(512,),	 'layer2.1.se_module.fc2.bias',	(512,),
 'block2.2.conv_bn1.conv.weight',	(256, 512, 1, 1),	 'layer2.2.conv1.weight',	(256, 512, 1, 1),
 'block2.2.conv_bn1.bn.weight',	(256,),	 'layer2.2.bn1.weight',	(256,),
 'block2.2.conv_bn1.bn.bias',	(256,),	 'layer2.2.bn1.bias',	(256,),
 'block2.2.conv_bn1.bn.running_mean',	(256,),	 'layer2.2.bn1.running_mean',	(256,),
 'block2.2.conv_bn1.bn.running_var',	(256,),	 'layer2.2.bn1.running_var',	(256,),
 'block2.2.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.2.conv2.weight',	(256, 8, 3, 3),
 'block2.2.conv_bn2.bn.weight',	(256,),	 'layer2.2.bn2.weight',	(256,),
 'block2.2.conv_bn2.bn.bias',	(256,),	 'layer2.2.bn2.bias',	(256,),
 'block2.2.conv_bn2.bn.running_mean',	(256,),	 'layer2.2.bn2.running_mean',	(256,),
 'block2.2.conv_bn2.bn.running_var',	(256,),	 'layer2.2.bn2.running_var',	(256,),
 'block2.2.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.2.conv3.weight',	(512, 256, 1, 1),
 'block2.2.conv_bn3.bn.weight',	(512,),	 'layer2.2.bn3.weight',	(512,),
 'block2.2.conv_bn3.bn.bias',	(512,),	 'layer2.2.bn3.bias',	(512,),
 'block2.2.conv_bn3.bn.running_mean',	(512,),	 'layer2.2.bn3.running_mean',	(512,),
 'block2.2.conv_bn3.bn.running_var',	(512,),	 'layer2.2.bn3.running_var',	(512,),
 'block2.2.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.2.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.2.scale.fc1.bias',	(32,),	 'layer2.2.se_module.fc1.bias',	(32,),
 'block2.2.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.2.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.2.scale.fc2.bias',	(512,),	 'layer2.2.se_module.fc2.bias',	(512,),
 'block2.3.conv_bn1.conv.weight',	(256, 512, 1, 1),	 'layer2.3.conv1.weight',	(256, 512, 1, 1),
 'block2.3.conv_bn1.bn.weight',	(256,),	 'layer2.3.bn1.weight',	(256,),
 'block2.3.conv_bn1.bn.bias',	(256,),	 'layer2.3.bn1.bias',	(256,),
 'block2.3.conv_bn1.bn.running_mean',	(256,),	 'layer2.3.bn1.running_mean',	(256,),
 'block2.3.conv_bn1.bn.running_var',	(256,),	 'layer2.3.bn1.running_var',	(256,),
 'block2.3.conv_bn2.conv.weight',	(256, 8, 3, 3),	 'layer2.3.conv2.weight',	(256, 8, 3, 3),
 'block2.3.conv_bn2.bn.weight',	(256,),	 'layer2.3.bn2.weight',	(256,),
 'block2.3.conv_bn2.bn.bias',	(256,),	 'layer2.3.bn2.bias',	(256,),
 'block2.3.conv_bn2.bn.running_mean',	(256,),	 'layer2.3.bn2.running_mean',	(256,),
 'block2.3.conv_bn2.bn.running_var',	(256,),	 'layer2.3.bn2.running_var',	(256,),
 'block2.3.conv_bn3.conv.weight',	(512, 256, 1, 1),	 'layer2.3.conv3.weight',	(512, 256, 1, 1),
 'block2.3.conv_bn3.bn.weight',	(512,),	 'layer2.3.bn3.weight',	(512,),
 'block2.3.conv_bn3.bn.bias',	(512,),	 'layer2.3.bn3.bias',	(512,),
 'block2.3.conv_bn3.bn.running_mean',	(512,),	 'layer2.3.bn3.running_mean',	(512,),
 'block2.3.conv_bn3.bn.running_var',	(512,),	 'layer2.3.bn3.running_var',	(512,),
 'block2.3.scale.fc1.weight',	(32, 512, 1, 1),	 'layer2.3.se_module.fc1.weight',	(32, 512, 1, 1),
 'block2.3.scale.fc1.bias',	(32,),	 'layer2.3.se_module.fc1.bias',	(32,),
 'block2.3.scale.fc2.weight',	(512, 32, 1, 1),	 'layer2.3.se_module.fc2.weight',	(512, 32, 1, 1),
 'block2.3.scale.fc2.bias',	(512,),	 'layer2.3.se_module.fc2.bias',	(512,),
 'block3.0.conv_bn1.conv.weight',	(512, 512, 1, 1),	 'layer3.0.conv1.weight',	(512, 512, 1, 1),
 'block3.0.conv_bn1.bn.weight',	(512,),	 'layer3.0.bn1.weight',	(512,),
 'block3.0.conv_bn1.bn.bias',	(512,),	 'layer3.0.bn1.bias',	(512,),
 'block3.0.conv_bn1.bn.running_mean',	(512,),	 'layer3.0.bn1.running_mean',	(512,),
 'block3.0.conv_bn1.bn.running_var',	(512,),	 'layer3.0.bn1.running_var',	(512,),
 'block3.0.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.0.conv2.weight',	(512, 16, 3, 3),
 'block3.0.conv_bn2.bn.weight',	(512,),	 'layer3.0.bn2.weight',	(512,),
 'block3.0.conv_bn2.bn.bias',	(512,),	 'layer3.0.bn2.bias',	(512,),
 'block3.0.conv_bn2.bn.running_mean',	(512,),	 'layer3.0.bn2.running_mean',	(512,),
 'block3.0.conv_bn2.bn.running_var',	(512,),	 'layer3.0.bn2.running_var',	(512,),
 'block3.0.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.0.conv3.weight',	(1024, 512, 1, 1),
 'block3.0.conv_bn3.bn.weight',	(1024,),	 'layer3.0.bn3.weight',	(1024,),
 'block3.0.conv_bn3.bn.bias',	(1024,),	 'layer3.0.bn3.bias',	(1024,),
 'block3.0.conv_bn3.bn.running_mean',	(1024,),	 'layer3.0.bn3.running_mean',	(1024,),
 'block3.0.conv_bn3.bn.running_var',	(1024,),	 'layer3.0.bn3.running_var',	(1024,),
 'block3.0.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.0.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.0.scale.fc1.bias',	(64,),	 'layer3.0.se_module.fc1.bias',	(64,),
 'block3.0.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.0.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.0.scale.fc2.bias',	(1024,),	 'layer3.0.se_module.fc2.bias',	(1024,),
 'block3.0.shortcut.conv.weight',	(1024, 512, 1, 1),	 'layer3.0.downsample.0.weight',	(1024, 512, 1, 1),
 'block3.0.shortcut.bn.weight',	(1024,),	 'layer3.0.downsample.1.weight',	(1024,),
 'block3.0.shortcut.bn.bias',	(1024,),	 'layer3.0.downsample.1.bias',	(1024,),
 'block3.0.shortcut.bn.running_mean',	(1024,),	 'layer3.0.downsample.1.running_mean',	(1024,),
 'block3.0.shortcut.bn.running_var',	(1024,),	 'layer3.0.downsample.1.running_var',	(1024,),
 'block3.1.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.1.conv1.weight',	(512, 1024, 1, 1),
 'block3.1.conv_bn1.bn.weight',	(512,),	 'layer3.1.bn1.weight',	(512,),
 'block3.1.conv_bn1.bn.bias',	(512,),	 'layer3.1.bn1.bias',	(512,),
 'block3.1.conv_bn1.bn.running_mean',	(512,),	 'layer3.1.bn1.running_mean',	(512,),
 'block3.1.conv_bn1.bn.running_var',	(512,),	 'layer3.1.bn1.running_var',	(512,),
 'block3.1.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.1.conv2.weight',	(512, 16, 3, 3),
 'block3.1.conv_bn2.bn.weight',	(512,),	 'layer3.1.bn2.weight',	(512,),
 'block3.1.conv_bn2.bn.bias',	(512,),	 'layer3.1.bn2.bias',	(512,),
 'block3.1.conv_bn2.bn.running_mean',	(512,),	 'layer3.1.bn2.running_mean',	(512,),
 'block3.1.conv_bn2.bn.running_var',	(512,),	 'layer3.1.bn2.running_var',	(512,),
 'block3.1.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.1.conv3.weight',	(1024, 512, 1, 1),
 'block3.1.conv_bn3.bn.weight',	(1024,),	 'layer3.1.bn3.weight',	(1024,),
 'block3.1.conv_bn3.bn.bias',	(1024,),	 'layer3.1.bn3.bias',	(1024,),
 'block3.1.conv_bn3.bn.running_mean',	(1024,),	 'layer3.1.bn3.running_mean',	(1024,),
 'block3.1.conv_bn3.bn.running_var',	(1024,),	 'layer3.1.bn3.running_var',	(1024,),
 'block3.1.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.1.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.1.scale.fc1.bias',	(64,),	 'layer3.1.se_module.fc1.bias',	(64,),
 'block3.1.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.1.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.1.scale.fc2.bias',	(1024,),	 'layer3.1.se_module.fc2.bias',	(1024,),
 'block3.2.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.2.conv1.weight',	(512, 1024, 1, 1),
 'block3.2.conv_bn1.bn.weight',	(512,),	 'layer3.2.bn1.weight',	(512,),
 'block3.2.conv_bn1.bn.bias',	(512,),	 'layer3.2.bn1.bias',	(512,),
 'block3.2.conv_bn1.bn.running_mean',	(512,),	 'layer3.2.bn1.running_mean',	(512,),
 'block3.2.conv_bn1.bn.running_var',	(512,),	 'layer3.2.bn1.running_var',	(512,),
 'block3.2.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.2.conv2.weight',	(512, 16, 3, 3),
 'block3.2.conv_bn2.bn.weight',	(512,),	 'layer3.2.bn2.weight',	(512,),
 'block3.2.conv_bn2.bn.bias',	(512,),	 'layer3.2.bn2.bias',	(512,),
 'block3.2.conv_bn2.bn.running_mean',	(512,),	 'layer3.2.bn2.running_mean',	(512,),
 'block3.2.conv_bn2.bn.running_var',	(512,),	 'layer3.2.bn2.running_var',	(512,),
 'block3.2.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.2.conv3.weight',	(1024, 512, 1, 1),
 'block3.2.conv_bn3.bn.weight',	(1024,),	 'layer3.2.bn3.weight',	(1024,),
 'block3.2.conv_bn3.bn.bias',	(1024,),	 'layer3.2.bn3.bias',	(1024,),
 'block3.2.conv_bn3.bn.running_mean',	(1024,),	 'layer3.2.bn3.running_mean',	(1024,),
 'block3.2.conv_bn3.bn.running_var',	(1024,),	 'layer3.2.bn3.running_var',	(1024,),
 'block3.2.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.2.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.2.scale.fc1.bias',	(64,),	 'layer3.2.se_module.fc1.bias',	(64,),
 'block3.2.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.2.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.2.scale.fc2.bias',	(1024,),	 'layer3.2.se_module.fc2.bias',	(1024,),
 'block3.3.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.3.conv1.weight',	(512, 1024, 1, 1),
 'block3.3.conv_bn1.bn.weight',	(512,),	 'layer3.3.bn1.weight',	(512,),
 'block3.3.conv_bn1.bn.bias',	(512,),	 'layer3.3.bn1.bias',	(512,),
 'block3.3.conv_bn1.bn.running_mean',	(512,),	 'layer3.3.bn1.running_mean',	(512,),
 'block3.3.conv_bn1.bn.running_var',	(512,),	 'layer3.3.bn1.running_var',	(512,),
 'block3.3.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.3.conv2.weight',	(512, 16, 3, 3),
 'block3.3.conv_bn2.bn.weight',	(512,),	 'layer3.3.bn2.weight',	(512,),
 'block3.3.conv_bn2.bn.bias',	(512,),	 'layer3.3.bn2.bias',	(512,),
 'block3.3.conv_bn2.bn.running_mean',	(512,),	 'layer3.3.bn2.running_mean',	(512,),
 'block3.3.conv_bn2.bn.running_var',	(512,),	 'layer3.3.bn2.running_var',	(512,),
 'block3.3.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.3.conv3.weight',	(1024, 512, 1, 1),
 'block3.3.conv_bn3.bn.weight',	(1024,),	 'layer3.3.bn3.weight',	(1024,),
 'block3.3.conv_bn3.bn.bias',	(1024,),	 'layer3.3.bn3.bias',	(1024,),
 'block3.3.conv_bn3.bn.running_mean',	(1024,),	 'layer3.3.bn3.running_mean',	(1024,),
 'block3.3.conv_bn3.bn.running_var',	(1024,),	 'layer3.3.bn3.running_var',	(1024,),
 'block3.3.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.3.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.3.scale.fc1.bias',	(64,),	 'layer3.3.se_module.fc1.bias',	(64,),
 'block3.3.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.3.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.3.scale.fc2.bias',	(1024,),	 'layer3.3.se_module.fc2.bias',	(1024,),
 'block3.4.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.4.conv1.weight',	(512, 1024, 1, 1),
 'block3.4.conv_bn1.bn.weight',	(512,),	 'layer3.4.bn1.weight',	(512,),
 'block3.4.conv_bn1.bn.bias',	(512,),	 'layer3.4.bn1.bias',	(512,),
 'block3.4.conv_bn1.bn.running_mean',	(512,),	 'layer3.4.bn1.running_mean',	(512,),
 'block3.4.conv_bn1.bn.running_var',	(512,),	 'layer3.4.bn1.running_var',	(512,),
 'block3.4.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.4.conv2.weight',	(512, 16, 3, 3),
 'block3.4.conv_bn2.bn.weight',	(512,),	 'layer3.4.bn2.weight',	(512,),
 'block3.4.conv_bn2.bn.bias',	(512,),	 'layer3.4.bn2.bias',	(512,),
 'block3.4.conv_bn2.bn.running_mean',	(512,),	 'layer3.4.bn2.running_mean',	(512,),
 'block3.4.conv_bn2.bn.running_var',	(512,),	 'layer3.4.bn2.running_var',	(512,),
 'block3.4.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.4.conv3.weight',	(1024, 512, 1, 1),
 'block3.4.conv_bn3.bn.weight',	(1024,),	 'layer3.4.bn3.weight',	(1024,),
 'block3.4.conv_bn3.bn.bias',	(1024,),	 'layer3.4.bn3.bias',	(1024,),
 'block3.4.conv_bn3.bn.running_mean',	(1024,),	 'layer3.4.bn3.running_mean',	(1024,),
 'block3.4.conv_bn3.bn.running_var',	(1024,),	 'layer3.4.bn3.running_var',	(1024,),
 'block3.4.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.4.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.4.scale.fc1.bias',	(64,),	 'layer3.4.se_module.fc1.bias',	(64,),
 'block3.4.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.4.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.4.scale.fc2.bias',	(1024,),	 'layer3.4.se_module.fc2.bias',	(1024,),
 'block3.5.conv_bn1.conv.weight',	(512, 1024, 1, 1),	 'layer3.5.conv1.weight',	(512, 1024, 1, 1),
 'block3.5.conv_bn1.bn.weight',	(512,),	 'layer3.5.bn1.weight',	(512,),
 'block3.5.conv_bn1.bn.bias',	(512,),	 'layer3.5.bn1.bias',	(512,),
 'block3.5.conv_bn1.bn.running_mean',	(512,),	 'layer3.5.bn1.running_mean',	(512,),
 'block3.5.conv_bn1.bn.running_var',	(512,),	 'layer3.5.bn1.running_var',	(512,),
 'block3.5.conv_bn2.conv.weight',	(512, 16, 3, 3),	 'layer3.5.conv2.weight',	(512, 16, 3, 3),
 'block3.5.conv_bn2.bn.weight',	(512,),	 'layer3.5.bn2.weight',	(512,),
 'block3.5.conv_bn2.bn.bias',	(512,),	 'layer3.5.bn2.bias',	(512,),
 'block3.5.conv_bn2.bn.running_mean',	(512,),	 'layer3.5.bn2.running_mean',	(512,),
 'block3.5.conv_bn2.bn.running_var',	(512,),	 'layer3.5.bn2.running_var',	(512,),
 'block3.5.conv_bn3.conv.weight',	(1024, 512, 1, 1),	 'layer3.5.conv3.weight',	(1024, 512, 1, 1),
 'block3.5.conv_bn3.bn.weight',	(1024,),	 'layer3.5.bn3.weight',	(1024,),
 'block3.5.conv_bn3.bn.bias',	(1024,),	 'layer3.5.bn3.bias',	(1024,),
 'block3.5.conv_bn3.bn.running_mean',	(1024,),	 'layer3.5.bn3.running_mean',	(1024,),
 'block3.5.conv_bn3.bn.running_var',	(1024,),	 'layer3.5.bn3.running_var',	(1024,),
 'block3.5.scale.fc1.weight',	(64, 1024, 1, 1),	 'layer3.5.se_module.fc1.weight',	(64, 1024, 1, 1),
 'block3.5.scale.fc1.bias',	(64,),	 'layer3.5.se_module.fc1.bias',	(64,),
 'block3.5.scale.fc2.weight',	(1024, 64, 1, 1),	 'layer3.5.se_module.fc2.weight',	(1024, 64, 1, 1),
 'block3.5.scale.fc2.bias',	(1024,),	 'layer3.5.se_module.fc2.bias',	(1024,),
 'block4.0.conv_bn1.conv.weight',	(1024, 1024, 1, 1),	 'layer4.0.conv1.weight',	(1024, 1024, 1, 1),
 'block4.0.conv_bn1.bn.weight',	(1024,),	 'layer4.0.bn1.weight',	(1024,),
 'block4.0.conv_bn1.bn.bias',	(1024,),	 'layer4.0.bn1.bias',	(1024,),
 'block4.0.conv_bn1.bn.running_mean',	(1024,),	 'layer4.0.bn1.running_mean',	(1024,),
 'block4.0.conv_bn1.bn.running_var',	(1024,),	 'layer4.0.bn1.running_var',	(1024,),
 'block4.0.conv_bn2.conv.weight',	(1024, 32, 3, 3),	 'layer4.0.conv2.weight',	(1024, 32, 3, 3),
 'block4.0.conv_bn2.bn.weight',	(1024,),	 'layer4.0.bn2.weight',	(1024,),
 'block4.0.conv_bn2.bn.bias',	(1024,),	 'layer4.0.bn2.bias',	(1024,),
 'block4.0.conv_bn2.bn.running_mean',	(1024,),	 'layer4.0.bn2.running_mean',	(1024,),
 'block4.0.conv_bn2.bn.running_var',	(1024,),	 'layer4.0.bn2.running_var',	(1024,),
 'block4.0.conv_bn3.conv.weight',	(2048, 1024, 1, 1),	 'layer4.0.conv3.weight',	(2048, 1024, 1, 1),
 'block4.0.conv_bn3.bn.weight',	(2048,),	 'layer4.0.bn3.weight',	(2048,),
 'block4.0.conv_bn3.bn.bias',	(2048,),	 'layer4.0.bn3.bias',	(2048,),
 'block4.0.conv_bn3.bn.running_mean',	(2048,),	 'layer4.0.bn3.running_mean',	(2048,),
 'block4.0.conv_bn3.bn.running_var',	(2048,),	 'layer4.0.bn3.running_var',	(2048,),
 'block4.0.scale.fc1.weight',	(128, 2048, 1, 1),	 'layer4.0.se_module.fc1.weight',	(128, 2048, 1, 1),
 'block4.0.scale.fc1.bias',	(128,),	 'layer4.0.se_module.fc1.bias',	(128,),
 'block4.0.scale.fc2.weight',	(2048, 128, 1, 1),	 'layer4.0.se_module.fc2.weight',	(2048, 128, 1, 1),
 'block4.0.scale.fc2.bias',	(2048,),	 'layer4.0.se_module.fc2.bias',	(2048,),
 'block4.0.shortcut.conv.weight',	(2048, 1024, 1, 1),	 'layer4.0.downsample.0.weight',	(2048, 1024, 1, 1),
 'block4.0.shortcut.bn.weight',	(2048,),	 'layer4.0.downsample.1.weight',	(2048,),
 'block4.0.shortcut.bn.bias',	(2048,),	 'layer4.0.downsample.1.bias',	(2048,),
 'block4.0.shortcut.bn.running_mean',	(2048,),	 'layer4.0.downsample.1.running_mean',	(2048,),
 'block4.0.shortcut.bn.running_var',	(2048,),	 'layer4.0.downsample.1.running_var',	(2048,),
 'block4.1.conv_bn1.conv.weight',	(1024, 2048, 1, 1),	 'layer4.1.conv1.weight',	(1024, 2048, 1, 1),
 'block4.1.conv_bn1.bn.weight',	(1024,),	 'layer4.1.bn1.weight',	(1024,),
 'block4.1.conv_bn1.bn.bias',	(1024,),	 'layer4.1.bn1.bias',	(1024,),
 'block4.1.conv_bn1.bn.running_mean',	(1024,),	 'layer4.1.bn1.running_mean',	(1024,),
 'block4.1.conv_bn1.bn.running_var',	(1024,),	 'layer4.1.bn1.running_var',	(1024,),
 'block4.1.conv_bn2.conv.weight',	(1024, 32, 3, 3),	 'layer4.1.conv2.weight',	(1024, 32, 3, 3),
 'block4.1.conv_bn2.bn.weight',	(1024,),	 'layer4.1.bn2.weight',	(1024,),
 'block4.1.conv_bn2.bn.bias',	(1024,),	 'layer4.1.bn2.bias',	(1024,),
 'block4.1.conv_bn2.bn.running_mean',	(1024,),	 'layer4.1.bn2.running_mean',	(1024,),
 'block4.1.conv_bn2.bn.running_var',	(1024,),	 'layer4.1.bn2.running_var',	(1024,),
 'block4.1.conv_bn3.conv.weight',	(2048, 1024, 1, 1),	 'layer4.1.conv3.weight',	(2048, 1024, 1, 1),
 'block4.1.conv_bn3.bn.weight',	(2048,),	 'layer4.1.bn3.weight',	(2048,),
 'block4.1.conv_bn3.bn.bias',	(2048,),	 'layer4.1.bn3.bias',	(2048,),
 'block4.1.conv_bn3.bn.running_mean',	(2048,),	 'layer4.1.bn3.running_mean',	(2048,),
 'block4.1.conv_bn3.bn.running_var',	(2048,),	 'layer4.1.bn3.running_var',	(2048,),
 'block4.1.scale.fc1.weight',	(128, 2048, 1, 1),	 'layer4.1.se_module.fc1.weight',	(128, 2048, 1, 1),
 'block4.1.scale.fc1.bias',	(128,),	 'layer4.1.se_module.fc1.bias',	(128,),
 'block4.1.scale.fc2.weight',	(2048, 128, 1, 1),	 'layer4.1.se_module.fc2.weight',	(2048, 128, 1, 1),
 'block4.1.scale.fc2.bias',	(2048,),	 'layer4.1.se_module.fc2.bias',	(2048,),
 'block4.2.conv_bn1.conv.weight',	(1024, 2048, 1, 1),	 'layer4.2.conv1.weight',	(1024, 2048, 1, 1),
 'block4.2.conv_bn1.bn.weight',	(1024,),	 'layer4.2.bn1.weight',	(1024,),
 'block4.2.conv_bn1.bn.bias',	(1024,),	 'layer4.2.bn1.bias',	(1024,),
 'block4.2.conv_bn1.bn.running_mean',	(1024,),	 'layer4.2.bn1.running_mean',	(1024,),
 'block4.2.conv_bn1.bn.running_var',	(1024,),	 'layer4.2.bn1.running_var',	(1024,),
 'block4.2.conv_bn2.conv.weight',	(1024, 32, 3, 3),	 'layer4.2.conv2.weight',	(1024, 32, 3, 3),
 'block4.2.conv_bn2.bn.weight',	(1024,),	 'layer4.2.bn2.weight',	(1024,),
 'block4.2.conv_bn2.bn.bias',	(1024,),	 'layer4.2.bn2.bias',	(1024,),
 'block4.2.conv_bn2.bn.running_mean',	(1024,),	 'layer4.2.bn2.running_mean',	(1024,),
 'block4.2.conv_bn2.bn.running_var',	(1024,),	 'layer4.2.bn2.running_var',	(1024,),
 'block4.2.conv_bn3.conv.weight',	(2048, 1024, 1, 1),	 'layer4.2.conv3.weight',	(2048, 1024, 1, 1),
 'block4.2.conv_bn3.bn.weight',	(2048,),	 'layer4.2.bn3.weight',	(2048,),
 'block4.2.conv_bn3.bn.bias',	(2048,),	 'layer4.2.bn3.bias',	(2048,),
 'block4.2.conv_bn3.bn.running_mean',	(2048,),	 'layer4.2.bn3.running_mean',	(2048,),
 'block4.2.conv_bn3.bn.running_var',	(2048,),	 'layer4.2.bn3.running_var',	(2048,),
 'block4.2.scale.fc1.weight',	(128, 2048, 1, 1),	 'layer4.2.se_module.fc1.weight',	(128, 2048, 1, 1),
 'block4.2.scale.fc1.bias',	(128,),	 'layer4.2.se_module.fc1.bias',	(128,),
 'block4.2.scale.fc2.weight',	(2048, 128, 1, 1),	 'layer4.2.se_module.fc2.weight',	(2048, 128, 1, 1),
 'block4.2.scale.fc2.bias',	(2048,),	 'layer4.2.se_module.fc2.bias',	(2048,),
 'logit.weight',	(1000, 1280),	 'last_linear.weight',	(1000, 2048),
 'logit.bias',	(1000,),	 'last_linear.bias',	(1000,),
]
PRETRAIN_FILE = '/root/share/data/pretrain_model/se_resnext50_32x4d-a260b3a4.pth'
def load_pretrain(net, skip=[], pretrain_file=PRETRAIN_FILE, conversion=CONVERSION, is_print=True):

    #raise NotImplementedError
    print('\tload pretrain_file: %s'%pretrain_file)

    #pretrain_state_dict = torch.load(pretrain_file)
    pretrain_state_dict = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
    state_dict = net.state_dict()

    i = 0
    conversion = np.array(CONVERSION).reshape(-1,4)
    for key,_,pretrain_key,_ in conversion:
        if any(s in key for s in
            ['.num_batches_tracked',]+skip):
            continue

        #print('\t\t',key)
        if is_print:
            print('\t\t','%-48s  %-24s  <---  %-32s  %-24s'%(
                key, str(state_dict[key].shape),
                pretrain_key, str(pretrain_state_dict[pretrain_key].shape),
            ))
        i = i+1

        state_dict[key] = pretrain_state_dict[pretrain_key]

    net.load_state_dict(state_dict)
    print('')
    print('len(pretrain_state_dict.keys()) = %d'%len(pretrain_state_dict.keys()))
    print('len(state_dict.keys())          = %d'%len(state_dict.keys()))
    print('loaded    = %d'%i)
    print('')



###############################################################################
class ConvBn2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self, in_channel, reduction=4, excite_size=-1):
        super(SqueezeExcite, self).__init__()
        self.excite_size=excite_size

        self.fc1 = nn.Conv2d(in_channel, in_channel//reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(in_channel//reduction, in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        #print(x.shape)

        if IS_GATHER_EXCITE:
            #print('IS_GATHER_EXCITE')
            s = F.avg_pool2d(x, kernel_size=self.excite_size)
        else:
            s = F.adaptive_avg_pool2d(x,1)

        s = self.fc1(s)
        s = F.relu(s, inplace=True)
        s = self.fc2(s)

        if IS_GATHER_EXCITE:
          s = F.interpolate(s, size=(x.shape[2],x.shape[3]), mode='nearest')

        x = x*torch.sigmoid(s)
        return x



#############  resnext50 pyramid feature net #######################################
# https://github.com/Hsuxu/ResNeXt/blob/master/models.py
# https://github.com/D-X-Y/ResNeXt-DenseNet/blob/master/models/resnext.py
# https://github.com/miraclewkf/ResNeXt-PyTorch/blob/master/resnext.py


# bottleneck type C
class SENextBottleneckBlock(nn.Module):
    def __init__(self, in_channel, channel, out_channel, stride=1, group=32, reduction=16, excite_size=-1, is_shortcut=False):
        super(SENextBottleneckBlock, self).__init__()
        self.is_shortcut = is_shortcut

        self.conv_bn1 = ConvBn2d(in_channel,     channel, kernel_size=1, padding=0, stride=1)
        self.conv_bn2 = ConvBn2d(   channel,     channel, kernel_size=3, padding=1, stride=stride, groups=group)
        self.conv_bn3 = ConvBn2d(   channel, out_channel, kernel_size=1, padding=0, stride=1)
        self.scale    = SqueezeExcite(out_channel, reduction, excite_size)

        if is_shortcut:
            self.shortcut = ConvBn2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)


    def forward(self, x):
        z = F.relu(self.conv_bn1(x),inplace=True)
        z = F.relu(self.conv_bn2(z),inplace=True)
        z = self.scale(self.conv_bn3(z))

        if self.is_shortcut:
            z += self.shortcut(x)
        else:
            z += x

        z = F.relu(z,inplace=True)
        return z



def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)



def make_layer_c(in_planes, planes, out_planes, groups, num_blocks, stride):
    layers = []
    layers.append(SENextBottleneckBlock(in_planes, planes, out_planes, groups, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(SENextBottleneckBlock(out_planes, planes, out_planes, groups))

    return nn.Sequential(*layers)



#resnext50_32x4d
class ResNext50(nn.Module):

    def __init__(self, num_class=1000 ):
        super(ResNext50, self).__init__()


        self.block0  = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )


        self.block1  = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=0, stride=2, ceil_mode=True),
             SENextBottleneckBlock( 64, 128, 256, stride=1, is_shortcut=True,  excite_size=64),
          * [SENextBottleneckBlock(256, 128, 256, stride=1, is_shortcut=False, excite_size=64) for i in range(1,3)],
        )

        self.block2  = nn.Sequential(
             SENextBottleneckBlock(256, 256, 512, stride=2, is_shortcut=True,  excite_size=32),
          * [SENextBottleneckBlock(512, 256, 512, stride=1, is_shortcut=False, excite_size=32) for i in range(1,4)],
        )

        self.block3  = nn.Sequential(
             SENextBottleneckBlock( 512,512,1024, stride=2, is_shortcut=True,  excite_size=16),
          * [SENextBottleneckBlock(1024,512,1024, stride=1, is_shortcut=False, excite_size=16) for i in range(1,6)],
        )

        self.block4 = nn.Sequential(
             SENextBottleneckBlock(1024,1024,2048, stride=2, is_shortcut=True,  excite_size=8),
          * [SENextBottleneckBlock(2048,1024,2048, stride=1, is_shortcut=False, excite_size=8) for i in range(1,3)],
        )


        self.logit = nn.Linear(2048,num_class)



    def forward(self, x):
        batch_size = len(x)

        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        logit = self.logit(x)
        return logit



# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    net = ResNext50()
    load_pretrain(net, is_print=True)

    #---
    if 0:
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
        exit(0)

    #---
    if 1:
        IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
        IMAGE_RGB_STD  = [0.229, 0.224, 0.225]

        net = net.cuda().eval()

        image_dir ='/root/share/data/imagenet/dummy/256x256'
        for f in [
            'great_white_shark','screwdriver','ostrich','blad_eagle','english_foxhound','goldfish',
        ]:
            image_file = image_dir +'/%s.jpg'%f
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            #image = cv2.resize(image,dsize=(224,224))
            #image = image[16:16+224,16:16+224]

            image = image[:,:,::-1]
            image = (image.astype(np.float32)/255 -IMAGE_RGB_MEAN)/IMAGE_RGB_STD
            input = image.transpose(2,0,1)

            input = torch.from_numpy(input).float().cuda().unsqueeze(0)

            logit = net(input)
            proability = F.softmax(logit,-1)

            probability = proability.data.cpu().numpy().reshape(-1)
            argsort = np.argsort(-probability)

            print(f, image.shape)
            print(probability[:5])
            for t in range(5):
                print(t, argsort[t], probability[argsort[t]])
            print('')

            pass

    print('\nsucess!')

##############################################################################3
