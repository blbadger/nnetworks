# latent_space.py

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import seaborn as sns
import sklearn.decomposition as decomp
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from google.colab import files
from google.colab import drive

drive.mount('/content/gdrive')

data_dir = pathlib.Path('/content/gdrive/My Drive/googlenet',  fname='Combined')
image_count = len(list(data_dir.glob('*.png')))

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, image_type='.png'):
        self.image_name_ls = list(img_dir.glob('*' + image_type))
        self.img_labels = [item.name for item in data_dir.glob('*')]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints 
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.Resize(size=[299, 299])(image)

        # assign label 
        label = os.path.basename(img_path)
        return image, label


class NewGoogleNet(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.conv3.bn.apply(blank_batchnorm)
        # self.model.inception5a.branch1.bn.apply(blank_batchnorm)
        # self.model.inception5a.branch2[1].bn.apply(blank_batchnorm)
        # self.model.inception5a.branch3[1].bn.apply(blank_batchnorm)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.model.conv1(x)
        # N x 64 x 112 x 112
        x = self.model.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.model.conv2(x)
        # N x 64 x 56 x 56
        x = self.model.conv3(x)
        # # N x 192 x 56 x 56
        # x = self.model.maxpool2(x)
        # # N x 192 x 28 x 28
        # x = self.model.inception3a(x)
        # # N x 256 x 28 x 28
        # x = self.model.inception3b(x)
        # # N x 480 x 28 x 28
        # x = self.model.maxpool3(x)
        # # N x 480 x 14 x 14
        # x = self.model.inception4a(x)
        # # N x 512 x 14 x 14
        # x = self.model.inception4b(x)
        # # N x 512 x 14 x 14
        # x = self.model.inception4c(x)
        # # N x 512 x 14 x 14
        # x = self.model.inception4d(x)
        # # N x 528 x 14 x 14
        # x = self.model.inception4e(x)
        # # N x 832 x 14 x 14
        # x = self.model.maxpool4(x)
        # # N x 832 x 7 x 7
        # x = self.model.inception5a(x)
        # N x 832 x 7 x 7
        # x = self.model.inception5b(x)
        # # N x 1024 x 7 x 7
        # x = self.model.avgpool(x)
        # # N x 1024 x 1 x 1
        # x = torch.flatten(x, 1)
        # # N x 1024
        # x = self.model.dropout(x)
        # x = self.model.fc(x)
        # N x 1000 (num_classes)
        return x


def blank_batchnorm(layer):
    layer.reset_parameters()
    layer.eval()
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.zero_()
    return

googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True).to(device)
images = ImageDataset(data_dir, image_type='.png')

# network = NewGoogleNet(googlenet).to(device)
outputs, labels_arr = [], []
for i, image in enumerate(images):
    print (i)
    label = image[1]
    image = image[0].reshape(1, 3, 299, 299).to(device)
    output = googlenet(image)
    output = output.detach().cpu().numpy()
    outputs.append(output)
    i = 11
    while label[i] not in ',.':
        i += 1
    labels_arr.append(label[11:i])

outputs = torch.tensor(outputs)
outputs = outputs.reshape(len(outputs), 1000)
pca = decomp.PCA(n_components=2)
pca.fit(outputs)
print (pca.explained_variance_ratio_)
arr = pca.transform(outputs)
x, y = [i[0] for i in arr], [i[1] for i in arr]
plt.figure(figsize=(18, 18))
plt.scatter(x, y)
for i, label in enumerate(labels_arr):
    plt.annotate(label, (x[i], y[i]))

plt.xlabel('Feature 0')
plt.ylabel('Feature 4')
plt.title('GoogleNet Layer 5a Embedding')
plt.show()
plt.close()

sns.jointplot(x, y)
plt.show()
plt.close()

[['tench', 'goldfish', 18.96853], ['goldfish', 'flamingo', 9.068674], ['great white shark', 'tiger shark', 15.323574], ['tiger shark', 'sturgeon', 15.0638075], ['hammerhead', 'hammer', 8.967533], ['electric ray', 'stingray', 24.117239], ['stingray', 'electric ray', 22.915136], ['cock', 'hen', 14.836816], ['hen', 'cock', 12.271047], ['ostrich', 'bustard', 10.593788], ['brambling', 'junco', 10.1835985], ['goldfinch', 'house finch', 13.860473], ['house finch', 'brambling', 13.283669], ['junco', 'brambling', 13.774601], ['indigo bunting', 'bubble', 8.921491], ['robin', 'house finch', 12.170648], ['bulbul', 'quail', 14.352588], ['jay', 'indigo bunting', 12.402643], ['magpie', 'black grouse', 4.75366], ['chickadee', 'carton', 25.059637], ['water ouzel', 'red-backed sandpiper', 10.450129], ['kite', 'African grey', 9.247639], ['bald eagle', 'vulture', 10.956814], ['vulture', 'bald eagle', 12.517682], ['great grey owl', 'prairie chicken', 7.1483707], ['European fire salamander', 'common newt', 11.625909], ['common newt', 'eft', 14.502185], ['eft', 'common newt', 16.330286], ['spotted salamander', 'common newt', 11.431958], ['axolotl', 'banded gecko', 15.683359], ['bullfrog', 'tailed frog', 19.469316], ['tree frog', 'tailed frog', 15.345655], ['tailed frog', 'bullfrog', 18.25918], ['loggerhead', 'leatherback turtle', 20.685747], ['leatherback turtle', 'loggerhead', 21.627678], ['mud turtle', 'terrapin', 17.891914], ['terrapin', 'loggerhead', 17.39199], ['box turtle', 'terrapin', 16.937971], ['banded gecko', 'whiptail', 10.891773], ['common iguana', 'African chameleon', 12.964566], ['American chameleon', 'green lizard', 22.047796], ['whiptail', 'banded gecko', 14.941311], ['agama', 'banded gecko', 14.858381], ['frilled lizard', 'African chameleon', 12.371843], ['alligator lizard', 'banded gecko', 17.491032], ['Gila monster', 'banded gecko', 10.660124], ['green lizard', 'American chameleon', 19.588467], ['African chameleon', 'green lizard', 12.577893], ['Komodo dragon', 'common iguana', 15.860219], ['African crocodile', 'American alligator', 25.00694], ['American alligator', 'African crocodile', 22.45569], ['triceratops', 'carousel', 8.481228], ['thunder snake', 'nematode', 11.749543], ['ringneck snake', 'water snake', 18.167631], ['hognose snake', 'water snake', 11.283992], ['green snake', 'green mamba', 12.325545], ['king snake', 'sea snake', 11.6868305], ['garter snake', 'water snake', 12.382689], ['water snake', 'sea snake', 13.794962], ['vine snake', 'green snake', 14.938642], ['night snake', 'water snake', 15.116936], ['boa constrictor', 'rock python', 15.161138], ['rock python', 'boa constrictor', 22.481215], ['Indian cobra', 'green mamba', 11.20762], ['green mamba', 'candle', 15.460296], ['sea snake', 'water snake', 17.260115], ['horned viper', 'sidewinder', 9.30719], ['diamondback', 'sidewinder', 14.029367], ['sidewinder', 'handkerchief', 6.6846895], ['trilobite', 'chiton', 11.9407215], ['harvestman', 'barn spider', 9.432067], ['scorpion', 'crayfish', 16.88235], ['black and gold garden spider', 'spider web', 16.075352], ['barn spider', 'garden spider', 13.227559], ['garden spider', 'barn spider', 15.336229], ['black widow', 'barn spider', 15.791279], ['tarantula', 'wolf spider', 10.532088], ['wolf spider', 'tarantula', 15.334156], ['tick', 'isopod', 11.876674], ['centipede', 'scorpion', 12.816544], ['black grouse', 'prairie chicken', 10.380647], ['ptarmigan', 'partridge', 10.97301], ['ruffed grouse', 'partridge', 15.315232], ['prairie chicken', 'black grouse', 11.796943], ['peacock', 'lacewing', 11.476101], ['quail', 'partridge', 12.6294155], ['partridge', 'ruffed grouse', 15.516251], ['African grey', 'terrapin', 11.326008], ['macaw', 'lorikeet', 8.860477], ['sulphur-crested cockatoo', 'spoonbill', 12.165933], ['lorikeet', 'macaw', 9.598573], ['coucal', 'bulbul', 11.341155], ['bee eater', 'indigo bunting', 11.409385], ['hornbill', 'toucan', 17.895693], ['hummingbird', 'red-backed sandpiper', 11.436776], ['jacamar', 'hummingbird', 9.849566], ['toucan', 'hornbill', 13.395099], ['drake', 'red-breasted merganser', 10.633899], ['red-breasted merganser', 'redshank', 11.257078], ['goose', 'dowitcher', 7.6817536], ['black swan', 'European gallinule', 10.999118], ['tusker', 'Indian elephant', 20.923672], ['echidna', 'porcupine', 12.130698], ['platypus', 'otter', 14.156729], ['wallaby', 'hare', 16.538647], ['koala', 'Madagascar cat', 12.223323], ['wombat', 'wood rabbit', 10.008893], ['jellyfish', 'bubble', 11.862449], ['sea anemone', 'anemone fish', 8.869169], ['brain coral', 'coral reef', 8.784076], ['flatworm', 'sea slug', 10.374899], ['nematode', 'jellyfish', 7.5006123], ['conch', 'sea cucumber', 11.890138], ['snail', 'slug', 23.912424], ['slug', 'snail', 18.870296], ['sea slug', 'flatworm', 9.182591], ['chiton', 'trilobite', 8.806483], ['chambered nautilus', 'sea anemone', 10.615652], ['Dungeness crab', 'king crab', 19.889105], ['rock crab', 'Dungeness crab', 20.039228], ['fiddler crab', 'rock crab', 14.042335], ['king crab', 'Dungeness crab', 18.759432], ['American lobster', 'crayfish', 26.594467], ['spiny lobster', 'crayfish', 24.390368], ['crayfish', 'American lobster', 20.427868], ['hermit crab', 'crayfish', 15.418229], ['isopod', 'scorpion', 14.029774], ['white stork', 'black stork', 15.442653], ['black stork', 'white stork', 14.436737], ['spoonbill', 'flamingo', 13.98884], ['flamingo', 'goldfish', 13.458399], ['little blue heron', 'crane', 12.577237], ['American egret', 'little blue heron', 13.625785], ['bittern', 'little blue heron', 11.551173], ['crane', 'flamingo', 15.148866], ['limpkin', 'black stork', 13.7057705], ['European gallinule', 'bubble', 10.591263], ['American coot', 'dowitcher', 9.086697], ['bustard', 'ostrich', 9.265211], ['ruddy turnstone', 'red-backed sandpiper', 14.970217], ['red-backed sandpiper', 'dowitcher', 13.770988], ['redshank', 'dowitcher', 14.782804], ['dowitcher', 'redshank', 13.96282], ['oystercatcher', 'redshank', 11.962713], ['pelican', 'spoonbill', 13.459152], ['king penguin', 'sea lion', 6.8520293], ['albatross', 'goose', 10.554666], ['grey whale', 'sturgeon', 11.733328], ['killer whale', 'black stork', 9.2324705], ['dugong', 'ice bear', 17.790722], ['sea lion', 'dugong', 13.56577], ['Chihuahua', 'toy terrier', 13.254365], ['Japanese spaniel', 'papillon', 9.387217], ['Maltese dog', 'Lhasa', 13.157652], ['Pekinese', 'Lhasa', 14.042983], ['Shih-Tzu', 'Lhasa', 16.230473], ['Blenheim spaniel', 'cocker spaniel', 10.6090555], ['papillon', 'Shetland sheepdog', 10.338802], ['toy terrier', 'Chihuahua', 11.363121], ['Rhodesian ridgeback', 'redbone', 10.2829075], ['Afghan hound', 'Saluki', 20.63708], ['basset', 'bloodhound', 16.034859], ['beagle', 'English foxhound', 13.716879], ['bloodhound', 'basset', 16.496862], ['bluetick', 'German short-haired pointer', 16.954874], ['black-and-tan coonhound', 'bloodhound', 17.653908], ['Walker hound', 'English foxhound', 15.414156], ['English foxhound', 'Walker hound', 7.6202474], ['redbone', 'bloodhound', 14.150872], ['borzoi', 'whippet', 16.08376], ['Irish wolfhound', 'Scottish deerhound', 16.631092], ['Italian greyhound', 'whippet', 21.321877], ['whippet', 'Saluki', 15.102482], ['Ibizan hound', 'basenji', 11.381943], ['Norwegian elkhound', 'Siberian husky', 11.717777], ['otterhound', 'bloodhound', 14.453808], ['Saluki', 'Afghan hound', 11.782093], ['Scottish deerhound', 'Irish wolfhound', 12.902606], ['Weimaraner', 'Bedlington terrier', 10.712852], ['Staffordshire bullterrier', 'American Staffordshire terrier', 20.275974], ['American Staffordshire terrier', 'Staffordshire bullterrier', 17.057999], ['Bedlington terrier', 'African grey', 13.186205], ['Border terrier', 'Irish terrier', 8.909785], ['Kerry blue terrier', 'Lakeland terrier', 12.1998825], ['Irish terrier', 'Airedale', 19.311377], ['Norfolk terrier', 'Norwich terrier', 11.80002], ['Norwich terrier', 'cairn', 13.456078], ['Yorkshire terrier', 'silky terrier', 11.264034], ['wire-haired fox terrier', 'Lakeland terrier', 10.5063305], ['Lakeland terrier', 'Airedale', 13.437834], ['Sealyham terrier', 'West Highland white terrier', 15.043685], ['Airedale', 'Lakeland terrier', 14.976443], ['cairn', 'West Highland white terrier', 20.731564], ['Australian terrier', 'Norwich terrier', 14.5642605], ['Dandie Dinmont', 'Sealyham terrier', 11.340257], ['Boston bull', 'French bulldog', 9.366329], ['miniature schnauzer', 'standard schnauzer', 17.635216], ['giant schnauzer', 'Scotch terrier', 13.076674], ['standard schnauzer', 'miniature schnauzer', 12.633991], ['Scotch terrier', 'West Highland white terrier', 9.82734], ['Tibetan terrier', 'carton', 14.154555], ['silky terrier', 'Yorkshire terrier', 15.856399], ['soft-coated wheaten terrier', 'Sealyham terrier', 9.356176], ['West Highland white terrier', 'Scotch terrier', 17.74779], ['Lhasa', 'Shih-Tzu', 15.285541], ['flat-coated retriever', 'curly-coated retriever', 8.139293], ['curly-coated retriever', 'Irish water spaniel', 14.786096], ['golden retriever', 'Labrador retriever', 13.588689], ['Labrador retriever', 'golden retriever', 15.763118], ['Chesapeake Bay retriever', 'vizsla', 11.886663], ['German short-haired pointer', 'pajama', 10.083553], ['vizsla', 'Chesapeake Bay retriever', 10.551332], ['English setter', 'German short-haired pointer', 10.849332], ['Irish setter', 'redbone', 12.489503], ['Gordon setter', 'black-and-tan coonhound', 12.4691925], ['Brittany spaniel', 'Ibizan hound', 10.480975], ['clumber', 'English setter', 15.318354], ['English springer', 'English setter', 15.378199], ['Welsh springer spaniel', 'English setter', 14.312752], ['cocker spaniel', 'clumber', 13.082953], ['Sussex spaniel', 'cocker spaniel', 10.897391], ['Irish water spaniel', 'curly-coated retriever', 10.60176], ['kuvasz', 'Great Pyrenees', 12.381916], ['schipperke', 'groenendael', 8.388123], ['groenendael', 'schipperke', 16.178183], ['malinois', 'German shepherd', 17.869432], ['briard', 'soft-coated wheaten terrier', 15.156134], ['kelpie', 'Ibizan hound', 12.061469], ['komondor', 'swab', 13.3360405], ['Old English sheepdog', 'komondor', 13.755064], ['Shetland sheepdog', 'collie', 15.998146], ['collie', 'Shetland sheepdog', 15.636794], ['Border collie', 'collie', 8.430101], ['Bouvier des Flandres', 'briard', 15.523498], ['Rottweiler', 'Doberman', 10.475668], ['German shepherd', 'malinois', 11.84816], ['Doberman', 'miniature pinscher', 10.088132], ['miniature pinscher', 'toy terrier', 16.642807], ['Greater Swiss Mountain dog', 'EntleBucher', 13.389681], ['Bernese mountain dog', 'Greater Swiss Mountain dog', 14.8799095], ['Appenzeller', 'EntleBucher', 12.128019], ['EntleBucher', 'Greater Swiss Mountain dog', 10.5337515], ['boxer', 'whippet', 11.891608], ['bull mastiff', 'boxer', 16.30393], ['Tibetan mastiff', 'Newfoundland', 7.458689], ['French bulldog', 'Boston bull', 9.243021], ['Great Dane', 'whippet', 12.765314], ['Saint Bernard', 'borzoi', 9.724411], ['Eskimo dog', 'Siberian husky', 11.487982], ['malamute', 'Siberian husky', 18.139338], ['Siberian husky', 'Eskimo dog', 17.76316], ['dalmatian', 'English setter', 11.684773], ['affenpinscher', 'Shih-Tzu', 9.67733], ['basenji', 'Ibizan hound', 14.254199], ['pug', 'bull mastiff', 15.327286], ['Leonberg', 'soft-coated wheaten terrier', 9.513095], ['Newfoundland', 'Tibetan mastiff', 10.973986], ['Great Pyrenees', 'kuvasz', 19.926332], ['Samoyed', 'white wolf', 17.716448], ['Pomeranian', 'chow', 12.371214], ['chow', 'Pomeranian', 11.674045], ['keeshond', 'Norwegian elkhound', 11.699245], ['Brabancon griffon', 'pug', 11.495286], ['Pembroke', 'Cardigan', 13.814369], ['Cardigan', 'Pembroke', 8.275639], ['toy poodle', 'miniature poodle', 21.249561], ['miniature poodle', 'standard poodle', 15.985348], ['standard poodle', 'miniature poodle', 16.840092], ['Mexican hairless', 'armadillo', 13.7823715], ['timber wolf', 'coyote', 16.901901], ['white wolf', 'Samoyed', 12.267539], ['red wolf', 'mashed potato', 13.397514], ['coyote', 'grey fox', 11.922016], ['dingo', 'dhole', 16.659004], ['dhole', 'coyote', 8.911489], ['African hunting dog', 'dhole', 9.575883], ['hyena', 'African hunting dog', 13.213883], ['red fox', 'kit fox', 15.77446], ['kit fox', 'red fox', 17.466316], ['Arctic fox', 'ice bear', 13.597992], ['grey fox', 'coyote', 16.08856], ['tabby', 'Egyptian cat', 16.815336], ['tiger cat', 'tiger', 16.774574], ['Persian cat', 'Pomeranian', 11.75084], ['Siamese cat', 'Egyptian cat', 14.328258], ['Egyptian cat', 'tiger cat', 15.872078], ['cougar', 'carton', 18.894787], ['lynx', 'cougar', 13.906376], ['leopard', 'cheetah', 19.158592], ['snow leopard', 'leopard', 15.177712], ['jaguar', 'leopard', 14.115265], ['lion', 'cheetah', 13.08162], ['tiger', 'tiger cat', 10.920845], ['cheetah', 'leopard', 11.042662], ['brown bear', 'ice bear', 14.308595], ['American black bear', 'ice bear', 14.32885], ['ice bear', 'dugong', 15.16534], ['sloth bear', 'American black bear', 13.423887], ['mongoose', 'meerkat', 12.077449], ['meerkat', 'mongoose', 14.777502], ['tiger beetle', 'bubble', 8.779219], ['ladybug', 'hip', 6.8030286], ['ground beetle', 'long-horned beetle', 11.334318], ['long-horned beetle', 'leaf beetle', 12.212389], ['leaf beetle', 'ladybug', 12.103423], ['dung beetle', 'tick', 11.106427], ['rhinoceros beetle', 'dung beetle', 12.045826], ['weevil', 'ant', 9.750348], ['fly', 'dragonfly', 12.3029375], ['bee', 'ant', 9.175281], ['ant', 'carton', 34.3703], ['grasshopper', 'cricket', 19.181435], ['cricket', 'grasshopper', 17.561071], ['walking stick', 'mantis', 11.75971], ['cockroach', 'crayfish', 10.580088], ['mantis', 'grasshopper', 14.624519], ['cicada', 'lacewing', 15.287191], ['leafhopper', 'lacewing', 13.667231], ['lacewing', 'nematode', 14.0694065], ['dragonfly', 'damselfly', 16.512447], ['damselfly', 'dragonfly', 22.642399], ['admiral', 'lycaenid', 12.803488], ['ringlet', 'sulphur butterfly', 11.735667], ['monarch', 'lionfish', 5.3187003], ['cabbage butterfly', 'sulphur butterfly', 14.927029], ['sulphur butterfly', 'lycaenid', 9.871691], ['lycaenid', 'ringlet', 10.445966], ['starfish', 'goldfish', 9.0954275], ['sea urchin', 'sea anemone', 7.809132], ['sea cucumber', 'starfish', 14.734419], ['wood rabbit', 'hare', 18.473118], ['hare', 'wood rabbit', 18.739166], ['Angora', 'carton', 13.209154], ['hamster', 'mousetrap', 10.960077], ['porcupine', 'echidna', 17.527164], ['fox squirrel', 'guillotine', 9.414678], ['marmot', 'beaver', 11.270038], ['beaver', 'platypus', 14.304405], ['guinea pig', 'hamster', 11.417444], ['sorrel', 'Arabian camel', 9.679851], ['zebra', 'prairie chicken', 9.459592], ['hog', 'wild boar', 12.52976], ['wild boar', 'hog', 19.611506], ['warthog', 'hog', 12.3753805], ['hippopotamus', 'chimpanzee', 11.007001], ['ox', 'oxcart', 19.997114], ['water buffalo', 'ox', 11.54489], ['bison', 'water buffalo', 11.836083], ['ram', 'bighorn', 17.668043], ['bighorn', 'ram', 17.326645], ['ibex', 'impala', 10.11014], ['hartebeest', 'impala', 13.795445], ['impala', 'gazelle', 14.183656], ['gazelle', 'impala', 14.731031], ['Arabian camel', 'llama', 10.070833], ['llama', 'Arabian camel', 12.786928], ['weasel', 'black-footed ferret', 19.223434], ['mink', 'weasel', 11.864296], ['polecat', 'black-footed ferret', 19.514212], ['black-footed ferret', 'weasel', 18.693913], ['otter', 'beaver', 14.896255], ['skunk', 'badger', 9.092725], ['badger', 'platypus', 9.426478], ['armadillo', 'American alligator', 17.025816], ['three-toed sloth', 'bathtub', 14.208599], ['orangutan', 'chimpanzee', 10.229923], ['gorilla', 'chimpanzee', 16.62046], ['chimpanzee', 'orangutan', 9.52664], ['gibbon', 'spider monkey', 11.076124], ['siamang', 'carton', 11.033104], ['guenon', 'patas', 17.302256], ['patas', 'proboscis monkey', 13.246342], ['baboon', 'macaque', 11.152584], ['macaque', 'baboon', 12.774447], ['langur', 'patas', 13.674894], ['colobus', 'langur', 11.178977], ['proboscis monkey', 'patas', 9.582734], ['marmoset', 'titi', 15.171636], ['capuchin', 'titi', 9.168477], ['howler monkey', 'capuchin', 10.794419], ['titi', 'marmoset', 17.265562], ['spider monkey', 'langur', 8.968607], ['squirrel monkey', 'titi', 12.561757], ['Madagascar cat', 'corkscrew', 10.001284], ['indri', 'Madagascar cat', 19.452812], ['Indian elephant', 'African elephant', 15.665048], ['African elephant', 'tusker', 17.524445], ['lesser panda', 'goldfish', 8.360738], ['giant panda', 'lesser panda', 9.282129], ['barracouta', 'gar', 15.089672], ['eel', 'gar', 14.012605], ['coho', 'gar', 17.630919], ['rock beauty', 'goldfish', 11.161562], ['anemone fish', 'sea anemone', 12.823548], ['sturgeon', 'gar', 15.862816], ['gar', 'barracouta', 18.852777], ['lionfish', 'sea anemone', 8.023019], ['puffer', 'lionfish', 10.072697], ['abacus', 'rubber eraser', 8.113448], ['abaya', 'cloak', 12.230658], ['academic gown', 'mortarboard', 24.145279], ['accordion', 'trilobite', 7.9787455], ['acoustic guitar', 'electric guitar', 18.088787], ['aircraft carrier', 'warplane', 9.189335], ['airliner', 'warplane', 18.01796], ['airship', 'parachute', 14.094031], ['altar', 'church', 8.08236], ['ambulance', 'umbrella', 10.942236], ['amphibian', 'tank', 8.470401], ['analog clock', 'wall clock', 21.630318], ['apiary', 'crate', 15.078493], ['apron', 'carton', 27.383512], ['ashcan', 'bucket', 12.534643], ['assault rifle', 'rifle', 17.600328], ['backpack', 'candle', 31.876236], ['bakery', 'confectionery', 11.130903], ['balance beam', 'parallel bars', 10.693296], ['balloon', 'parachute', 11.308222], ['ballpoint', 'syringe', 21.758667], ['Band Aid', 'envelope', 9.070374], ['banjo', 'electric guitar', 11.113703], ['bannister', 'coil', 9.494946], ['barbell', 'dumbbell', 19.287376], ['barber chair', 'barbershop', 14.737461], ['barbershop', 'barber chair', 12.934062], ['barn', 'birdhouse', 13.339354], ['barometer', 'stopwatch', 13.986485], ['barrel', 'rain barrel', 14.243166], ['barrow', 'tricycle', 10.944197], ['baseball', 'paper towel', 10.007555], ['basketball', 'volleyball', 12.067942], ['bassinet', 'cradle', 16.762047], ['bassoon', 'oboe', 15.06136], ['bathing cap', 'shower cap', 15.485221], ['bath towel', 'carton', 20.93111], ['bathtub', 'tub', 28.583832], ['beach wagon', 'jeep', 10.63764], ['beacon', 'church', 11.02064], ['beaker', 'measuring cup', 21.771957], ['bearskin', 'pickelhaube', 12.742806], ['beer bottle', 'pop bottle', 15.884791], ['beer glass', 'goblet', 15.08543], ['bell cote', 'church', 13.222763], ['bib', 'apron', 7.7023077], ['bicycle-built-for-two', 'unicycle', 13.947419], ['bikini', 'bathing cap', 11.641664], ['binder', 'wallet', 13.321364], ['binoculars', 'gasmask', 7.0456233], ['birdhouse', 'barn', 12.277554], ['boathouse', 'lakeside', 11.389854], ['bobsled', 'pop bottle', 8.711297], ['bolo tie', 'necklace', 12.541467], ['bonnet', 'shower cap', 20.92212], ['bookcase', 'library', 12.900629], ['bookshop', 'bookcase', 8.289765], ['bottlecap', 'necklace', 8.902738], ['bow', 'torch', 10.707405], ['bow tie', 'candle', 37.51272], ['brass', 'menu', 8.193935], ['brassiere', 'honeycomb', 11.573363], ['breakwater', 'dam', 7.112547], ['breastplate', 'cuirass', 22.246729], ['broom', 'swab', 27.380611], ['bucket', 'measuring cup', 15.95048], ['buckle', 'shield', 8.22209], ['bulletproof vest', 'military uniform', 16.225023], ['bullet train', 'warplane', 8.928124], ['butcher shop', 'shower curtain', 8.557744], ['cab', 'racer', 10.465584], ['caldron', 'milk can', 9.473121], ['candle', 'carton', 33.43836], ['cannon', 'projectile', 16.23065], ['canoe', 'paddle', 13.36611], ['can opener', 'corkscrew', 9.513902], ['cardigan', 'stole', 9.725083], ['car mirror', 'television', 7.594341], ['carousel', 'mosquito net', 6.453425], ["carpenter's kit", 'hammer', 10.338543], ['carton', 'candle', 34.851154], ['car wheel', 'disk brake', 5.6844606], ['cash machine', 'slot', 9.316392], ['cassette', 'tape player', 9.07442], ['cassette player', 'tape player', 14.145056], ['castle', 'jigsaw puzzle', 10.744833], ['catamaran', 'trimaran', 17.200308], ['CD player', 'tape player', 10.774914], ['cello', 'violin', 18.088675], ['cellular telephone', 'iPod', 12.317073], ['chain', 'swing', 13.216865], ['chainlink fence', 'window screen', 6.081715], ['chain mail', 'cuirass', 17.598589], ['chain saw', 'mousetrap', 7.7261596], ['chest', 'crate', 10.383025], ['chiffonier', 'chest', 9.910445], ['chime', 'rule', 8.762611], ['china cabinet', 'plate rack', 10.957293], ['Christmas stocking', 'sock', 11.574301], ['church', 'altar', 12.164408], ['cinema', 'theater curtain', 7.210852], ['cleaver', 'hatchet', 23.114056], ['cliff dwelling', 'cliff', 7.439912], ['cloak', 'abaya', 11.867491], ['clog', 'sandal', 16.958557], ['cocktail shaker', 'vase', 10.282954], ['coffee mug', 'cup', 19.770576], ['coffeepot', 'water jug', 19.56217], ['coil', 'chambered nautilus', 9.070422], ['combination lock', 'shower cap', 11.689611], ['computer keyboard', 'space bar', 10.820548], ['confectionery', 'toyshop', 10.31559], ['container ship', 'drilling platform', 9.22594], ['convertible', 'sports car', 12.500901], ['corkscrew', 'Madagascar cat', 9.5283575], ['cornet', 'trombone', 21.144247], ['cowboy boot', 'shoe shop', 17.056509], ['cowboy hat', 'sombrero', 33.94435], ['cradle', 'bassinet', 20.644396], ['crane', 'drilling platform', 12.79747], ['crash helmet', 'football helmet', 10.831262], ['crate', 'carton', 16.415398], ['crib', 'cradle', 17.692373], ['Crock Pot', 'toaster', 10.246116], ['croquet ball', 'maraca', 11.024205], ['crutch', 'swab', 14.364605], ['cuirass', 'breastplate', 26.836252], ['dam', 'fountain', 9.172857], ['desk', 'carton', 33.230885], ['desktop computer', 'monitor', 15.945111], ['dial telephone', 'iron', 9.221715], ['diaper', 'handkerchief', 10.181061], ['digital clock', 'digital watch', 12.520313], ['digital watch', 'stopwatch', 14.43319], ['dining table', 'tray', 8.514129], ['dishrag', 'handkerchief', 7.7782674], ['dishwasher', 'refrigerator', 14.131912], ['disk brake', 'oxygen mask', 7.0871544], ['dock', 'container ship', 10.583584], ['dogsled', 'Eskimo dog', 10.223294], ['dome', 'yurt', 10.558337], ['doormat', 'book jacket', 8.804607], ['drilling platform', 'container ship', 10.656417], ['drum', 'drumstick', 16.817503], ['drumstick', 'plunger', 18.22631], ['dumbbell', 'barbell', 20.337595], ['Dutch oven', 'frying pan', 9.495956], ['electric fan', 'oxygen mask', 10.390237], ['electric guitar', 'banjo', 16.005014], ['electric locomotive', 'freight car', 7.1212735], ['entertainment center', 'television', 13.406826], ['envelope', 'carton', 10.01542], ['espresso maker', 'coffeepot', 15.839733], ['face powder', 'Petri dish', 9.9563675], ['feather boa', 'rule', 14.745113], ['file', 'chiffonier', 12.14554], ['fireboat', 'fountain', 22.733513], ['fire engine', 'tow truck', 8.073247], ['fire screen', 'stove', 10.416022], ['flagpole', 'parachute', 8.28919], ['flute', 'oboe', 19.264074], ['folding chair', 'pinwheel', 5.9463916], ['football helmet', 'oxygen mask', 17.700651], ['forklift', 'crane', 9.098077], ['fountain', 'fireboat', 19.828857], ['fountain pen', 'ballpoint', 15.436992], ['four-poster', 'mosquito net', 16.923674], ['freight car', 'trailer truck', 9.909702], ['French horn', 'cornet', 13.585305], ['frying pan', 'wok', 10.874559], ['fur coat', 'cloak', 9.463485], ['garbage truck', 'harvester', 10.389636], ['gasmask', 'oxygen mask', 28.284113], ['gas pump', 'vending machine', 8.032274], ['goblet', 'red wine', 13.835287], ['go-kart', 'lawn mower', 10.914851], ['golf ball', 'Petri dish', 6.761778], ['golfcart', 'forklift', 9.156988], ['gondola', 'canoe', 6.873725], ['gong', 'chime', 19.412949], ['gown', 'hoopskirt', 17.740322], ['grand piano', 'upright', 14.611856], ['greenhouse', 'mosquito net', 8.580427], ['grille', 'bubble', 11.993043], ['grocery store', 'confectionery', 12.121941], ['guillotine', 'cassette', 16.217026], ['hair slide', 'necklace', 11.372976], ['hair spray', 'syringe', 22.34437], ['half track', 'tank', 9.4449835], ['hammer', 'hatchet', 23.440172], ['hamper', 'bassinet', 13.575676], ['hand blower', 'whistle', 16.142702], ['hand-held computer', 'screen', 9.534021], ['handkerchief', 'bath towel', 13.071132], ['hard disc', 'Greater Swiss Mountain dog', 8.092972], ['harmonica', 'ocarina', 8.15436], ['harp', 'crane', 13.895683], ['harvester', 'plow', 11.423584], ['hatchet', 'cleaver', 21.853289], ['holster', 'scabbard', 10.494672], ['home theater', 'television', 15.996799], ['honeycomb', 'apiary', 9.072665], ['hook', 'chain', 11.808484], ['hoopskirt', 'overskirt', 19.117962], ['horizontal bar', 'parallel bars', 13.090548], ['horse cart', 'oxcart', 13.587409], ['hourglass', 'organ', 15.843678], ['iPod', 'remote control', 10.788962], ['iron', 'dial telephone', 11.64598], ["jack-o'-lantern", 'mask', 7.138725], ['jean', 'miniskirt', 8.908716], ['jeep', 'tow truck', 7.602317], ['jersey', 'sweatshirt', 12.052127], ['jigsaw puzzle', 'envelope', 7.1585617], ['jinrikisha', 'tricycle', 9.218383], ['joystick', 'goblet', 7.702376], ['kimono', 'vestment', 11.732272], ['knee pad', 'sandal', 8.413293], ['knot', 'chain', 11.627084], ['lab coat', 'hourglass', 14.195257], ['ladle', 'wooden spoon', 25.985588], ['lampshade', 'table lamp', 20.690548], ['laptop', 'notebook', 18.361385], ['lawn mower', 'harvester', 9.32258], ['lens cap', 'bottlecap', 6.9878154], ['letter opener', 'spatula', 13.049705], ['library', 'bookshop', 9.86959], ['lifeboat', 'fireboat', 6.774026], ['lighter', 'candle', 20.570347], ['limousine', 'scorpion', 9.541705], ['liner', 'container ship', 13.611018], ['lipstick', 'candle', 14.978237], ['Loafer', 'sandal', 18.574104], ['lotion', 'sunscreen', 13.438926], ['loudspeaker', 'tape player', 7.5381646], ['loupe', 'bubble', 14.626289], ['lumbermill', 'thresher', 8.514708], ['magnetic compass', 'stopwatch', 8.725087], ['mailbag', 'purse', 14.405627], ['mailbox', 'chest', 13.229669], ['maillot', 'bikini', 13.78483], ['maillot', 'bathing cap', 12.859812], ['manhole cover', 'sundial', 7.399606], ['maraca', 'plunger', 14.887858], ['marimba', 'panpipe', 13.948136], ['mask', 'ski mask', 13.639997], ['matchstick', 'candle', 13.36833], ['maypole', 'stupa', 7.971556], ['maze', 'manhole cover', 9.850941], ['measuring cup', 'beaker', 15.216508], ['medicine chest', 'washbasin', 12.511725], ['megalith', 'stone wall', 5.5308075], ['microphone', 'hand blower', 17.477915], ['microwave', 'oscilloscope', 11.106055], ['military uniform', 'bulletproof vest', 18.663242], ['milk can', 'bucket', 11.5467205], ['minibus', 'ambulance', 9.866307], ['miniskirt', 'swimming trunks', 10.843819], ['minivan', 'minibus', 9.387359], ['missile', 'projectile', 23.18629], ['mitten', 'sock', 9.401717], ['mixing bowl', 'cup', 14.112426], ['mobile home', 'mosquito net', 7.1529155], ['Model T', 'tractor', 7.131881], ['modem', 'projector', 12.309271], ['monastery', 'church', 12.7030945], ['monitor', 'candle', 39.066837], ['moped', 'motor scooter', 14.918688], ['mortar', "potter's wheel", 13.034001], ['mortarboard', 'academic gown', 22.515842], ['mosque', 'dome', 11.127034], ['mosquito net', 'shower curtain', 11.5172825], ['motor scooter', 'carton', 18.348272], ['mountain bike', 'unicycle', 15.426266], ['mountain tent', 'umbrella', 19.31043], ['mouse', 'computer keyboard', 10.486986], ['mousetrap', 'scorpion', 8.768589], ['moving van', 'garbage truck', 9.219794], ['muzzle', 'oxygen mask', 22.664946], ['nail', 'screw', 13.91151], ['neck brace', 'iPod', 13.736492], ['necklace', 'chain', 10.598272], ['nipple', 'oxygen mask', 22.016043], ['notebook', 'laptop', 18.09735], ['obelisk', 'rule', 7.016497], ['oboe', 'flute', 16.605026], ['ocarina', 'birdhouse', 8.426237], ['odometer', 'analog clock', 11.965286], ['oil filter', 'pencil sharpener', 7.8079004], ['organ', 'hourglass', 17.421383], ['oscilloscope', 'digital clock', 11.878817], ['overskirt', 'hoopskirt', 18.737505], ['oxcart', 'ox', 15.395498], ['oxygen mask', 'nipple', 27.756357], ['packet', 'plastic bag', 17.33654], ['paddle', 'canoe', 11.315876], ['paddlewheel', 'shower cap', 13.332394], ['padlock', 'combination lock', 12.196556], ['paintbrush', 'broom', 21.200375], ['pajama', 'candle', 23.497063], ['palace', 'monastery', 7.624227], ['panpipe', 'comic book', 11.413043], ['paper towel', 'toilet tissue', 33.510857], ['parachute', 'umbrella', 13.180613], ['parallel bars', 'horizontal bar', 16.810364], ['park bench', 'sundial', 8.197186], ['parking meter', 'digital watch', 12.877573], ['passenger car', 'electric locomotive', 10.382439], ['patio', 'dining table', 10.322266], ['pay-phone', 'dial telephone', 14.818706], ['pedestal', 'sundial', 10.946975], ['pencil box', 'wallet', 13.037634], ['pencil sharpener', 'rubber eraser', 8.775071], ['perfume', 'lotion', 10.232009], ['Petri dish', 'bubble', 11.944214], ['photocopier', 'bulletproof vest', 15.281446], ['pick', 'viaduct', 12.486926], ['pickelhaube', 'bearskin', 9.173972], ['picket fence', 'worm fence', 11.369286], ['pickup', 'tow truck', 13.579241], ['pier', 'steel arch bridge', 13.707804], ['piggy bank', 'hog', 11.647847], ['pill bottle', 'nipple', 13.972378], ['pillow', 'quilt', 13.919712], ['ping-pong ball', 'golf ball', 14.390179], ['pinwheel', 'European gallinule', 6.366708], ['pirate', 'schooner', 13.971711], ['pitcher', 'water jug', 19.807074], ['plane', 'mousetrap', 11.394772], ['planetarium', 'dome', 15.397906], ['plastic bag', 'shower cap', 16.491488], ['plate rack', 'china cabinet', 13.275161], ['plow', 'jigsaw puzzle', 7.3877716], ['plunger', 'maraca', 15.155772], ['Polaroid camera', 'projector', 11.785095], ['pole', 'maypole', 11.962577], ['police van', 'ambulance', 12.865012], ['poncho', 'stole', 13.6669655], ['pool table', 'ping-pong ball', 11.153776], ['pop bottle', 'water bottle', 20.973913], ['pot', 'vase', 11.779526], ["potter's wheel", 'mortar', 9.429176], ['power drill', 'hand blower', 18.28285], ['prayer rug', 'jigsaw puzzle', 5.3223643], ['printer', 'photocopier', 17.635145], ['prison', 'vault', 9.261132], ['projectile', 'missile', 25.08661], ['projector', 'modem', 10.8182955], ['puck', 'ski', 7.295972], ['punching bag', 'chime', 15.335311], ['purse', 'bib', 10.832087], ['quill', 'broom', 13.206905], ['quilt', 'pillow', 12.3594265], ['racer', 'sports car', 10.442243], ['racket', 'tennis ball', 17.90446], ['radiator', 'space heater', 10.847821], ['radio', 'oscilloscope', 13.92868], ['radio telescope', 'solar dish', 15.985693], ['rain barrel', 'barrel', 14.066951], ['recreational vehicle', 'carton', 10.752228], ['reel', 'parachute', 7.7848487], ['reflex camera', 'gasmask', 10.381313], ['refrigerator', 'dishwasher', 9.75494], ['remote control', 'Band Aid', 9.489223], ['restaurant', 'dining table', 13.5341625], ['revolver', 'holster', 15.980306], ['rifle', 'assault rifle', 22.552643], ['rocking chair', 'folding chair', 8.8172865], ['rotisserie', 'American lobster', 9.18844], ['rubber eraser', 'candle', 30.006021], ['rugby ball', 'soccer ball', 6.9695773], ['rule', 'feather boa', 13.94734], ['running shoe', 'sandal', 14.134542], ['safe', 'carton', 31.58609], ['safety pin', 'measuring cup', 9.941889], ['saltshaker', 'thimble', 12.604805], ['sandal', 'candle', 22.389984], ['sarong', 'cicada', 11.236579], ['sax', 'cornet', 15.0141945], ['scabbard', 'hatchet', 9.608312], ['scale', 'magnetic compass', 13.084346], ['school bus', 'cab', 7.602802], ['schooner', 'pirate', 10.597083], ['scoreboard', 'menu', 9.634529], ['screen', 'monitor', 18.53551], ['screw', 'nail', 9.626206], ['screwdriver', 'syringe', 23.184406], ['seat belt', 'bassinet', 5.855525], ['sewing machine', 'damselfly', 12.821579], ['shield', 'breastplate', 12.418769], ['shoe shop', 'clog', 19.72806], ['shoji', 'window shade', 9.121731], ['shopping basket', 'carton', 27.316504], ['shopping cart', 'shopping basket', 15.008381], ['shovel', 'hatchet', 16.633318], ['shower cap', 'paddlewheel', 15.142477], ['shower curtain', 'mosquito net', 9.647846], ['ski', 'dogsled', 11.291839], ['ski mask', 'mask', 17.018505], ['sleeping bag', 'shower cap', 11.030204], ['slide rule', 'barometer', 11.019012], ['sliding door', 'china cabinet', 7.2209673], ['slot', 'cash machine', 8.651145], ['snorkel', 'oxygen mask', 14.663946], ['snowmobile', 'ski', 13.951252], ['snowplow', 'harvester', 6.884259], ['soap dispenser', 'washbasin', 17.04174], ['soccer ball', 'volleyball', 9.801536], ['sock', 'Christmas stocking', 9.291442], ['solar dish', 'radio telescope', 10.927111], ['sombrero', 'cowboy hat', 28.468952], ['soup bowl', 'consomme', 13.033616], ['space bar', 'computer keyboard', 15.872862], ['space heater', 'electric fan', 12.237113], ['space shuttle', 'missile', 14.8578825], ['spatula', 'wooden spoon', 16.325333], ['speedboat', 'fireboat', 8.336555], ['spider web', 'garden spider', 11.567901], ['spindle', 'plunger', 13.2730875], ['sports car', 'racer', 13.613367], ['spotlight', 'bubble', 13.156781], ['stage', 'toyshop', 7.9177337], ['steam locomotive', 'thresher', 7.1514473], ['steel arch bridge', 'pier', 11.625822], ['steel drum', 'Petri dish', 10.795546], ['stethoscope', 'bolo tie', 12.394489], ['stole', 'poncho', 12.137344], ['stone wall', 'bulletproof vest', 6.5204196], ['stopwatch', 'analog clock', 13.947317], ['stove', 'fire screen', 17.155312], ['strainer', 'measuring cup', 9.435483], ['streetcar', 'passenger car', 8.722744], ['stretcher', 'sleeping bag', 9.318502], ['studio couch', 'quilt', 13.181317], ['stupa', 'mashed potato', 9.860644], ['submarine', 'dam', 10.689648], ['suit', 'Windsor tie', 10.988237], ['sundial', 'magnetic compass', 12.235728], ['sunglass', 'sunglasses', 13.682996], ['sunglasses', 'sunglass', 15.215174], ['sunscreen', 'lotion', 12.362833], ['suspension bridge', 'pier', 11.755079], ['swab', 'broom', 26.194923], ['sweatshirt', 'jersey', 14.420015], ['swimming trunks', 'diaper', 12.633121], ['swing', 'chain', 12.568549], ['switch', 'safe', 12.428088], ['syringe', 'rule', 16.267431], ['table lamp', 'lampshade', 24.03825], ['tank', 'half track', 12.882402], ['tape player', 'cassette player', 12.353519], ['teapot', 'coffeepot', 20.172138], ['teddy', 'toyshop', 11.536886], ['television', 'screen', 19.940697], ['tennis ball', 'basketball', 13.42913], ['thatch', 'yurt', 9.583717], ['theater curtain', 'shower curtain', 6.577014], ['thimble', 'saltshaker', 12.197744], ['thresher', 'harvester', 11.255485], ['throne', 'altar', 11.004519], ['tile roof', 'radiator', 8.798149], ['toaster', 'rotisserie', 10.457817], ['tobacco shop', 'toyshop', 8.440584], ['toilet seat', 'plunger', 14.60842], ['torch', 'candle', 20.253756], ['totem pole', 'pole', 10.736407], ['tow truck', 'crane', 12.522545], ['toyshop', 'confectionery', 6.7515383], ['tractor', 'plow', 16.393656], ['trailer truck', 'garbage truck', 10.046049], ['tray', 'carton', 32.33872], ['trench coat', 'military uniform', 8.478201], ['tricycle', 'unicycle', 7.9640694], ['trimaran', 'catamaran', 13.256825], ['tripod', 'crane', 8.065235], ['triumphal arch', 'monastery', 11.1815195], ['trolleybus', 'streetcar', 11.275441], ['trombone', 'cornet', 15.543886], ['tub', 'bathtub', 28.295584], ['turnstile', 'measuring cup', 8.732084], ['typewriter keyboard', 'computer keyboard', 16.217148], ['umbrella', 'parachute', 14.233441], ['unicycle', 'mountain bike', 16.537884], ['upright', 'grand piano', 12.260202], ['vacuum', 'oxygen mask', 14.672387], ['vase', 'water jug', 16.88722], ['vault', 'nipple', 8.247825], ['velvet', 'handkerchief', 11.502223], ['vending machine', 'refrigerator', 8.20876], ['vestment', 'throne', 7.298624], ['viaduct', 'steel arch bridge', 8.40178], ['violin', 'cello', 17.04212], ['volleyball', 'basketball', 13.206557], ['waffle iron', 'manhole cover', 7.6903872], ['wall clock', 'analog clock', 20.877275], ['wallet', 'purse', 12.870164], ['wardrobe', 'shower curtain', 6.8556414], ['warplane', 'wing', 14.73857], ['washbasin', 'bathtub', 21.798542], ['washer', 'dishwasher', 9.556624], ['water bottle', 'nipple', 20.085732], ['water jug', 'pitcher', 23.069883], ['water tower', 'mosque', 10.212414], ['whiskey jug', 'pitcher', 13.672069], ['whistle', 'candle', 35.95157], ['wig', 'shower cap', 15.896104], ['window screen', 'shower curtain', 9.125807], ['window shade', 'shower curtain', 12.558048], ['Windsor tie', 'suit', 12.118081], ['wine bottle', 'red wine', 18.864882], ['wing', 'warplane', 11.430428], ['wok', 'frying pan', 12.25588], ['wooden spoon', 'ladle', 26.092113], ['wool', 'spindle', 9.7493515], ['worm fence', 'picket fence', 6.2121263], ['wreck', 'paddlewheel', 10.943399], ['yawl', 'catamaran', 13.485539], ['yurt', 'dome', 12.489111], ['web site', 'monitor', 9.851211], ['comic book', 'book jacket', 9.936756], ['crossword puzzle', 'jigsaw puzzle', 8.532192], ['street sign', 'traffic light', 6.3551607], ['traffic light', 'street sign', 6.0626087], ['book jacket', 'comic book', 9.876173], ['menu', 'measuring cup', 7.111881], ['plate', 'tray', 8.719539], ['guacamole', 'burrito', 6.7248363], ['consomme', 'soup bowl', 12.4470415], ['hot pot', 'caldron', 9.799569], ['trifle', 'bulletproof vest', 11.355886], ['ice cream', 'cauliflower', 12.50799], ['ice lolly', 'nipple', 15.276102], ['French loaf', 'Band Aid', 10.418798], ['bagel', 'pretzel', 15.824649], ['pretzel', 'nematode', 7.3479414], ['cheeseburger', 'guacamole', 11.890758], ['hotdog', 'cheeseburger', 8.7706785], ['mashed potato', 'cauliflower', 17.59965], ['head cabbage', 'cauliflower', 12.641164], ['broccoli', 'cauliflower', 13.492819], ['cauliflower', 'mashed potato', 16.568243], ['zucchini', 'cucumber', 19.177263], ['spaghetti squash', 'carbonara', 12.856659], ['acorn squash', 'spaghetti squash', 16.621855], ['butternut squash', 'acorn squash', 9.180733], ['cucumber', 'feather boa', 14.913129], ['artichoke', 'cardoon', 11.093243], ['bell pepper', 'Granny Smith', 9.276788], ['cardoon', 'artichoke', 11.115046], ['mushroom', 'agaric', 12.822686], ['Granny Smith', 'bagel', 8.881373], ['strawberry', 'goldfish', 6.613924], ['orange', 'shower cap', 17.530136], ['lemon', 'orange', 15.071481], ['fig', 'acorn squash', 9.733798], ['pineapple', 'cardoon', 7.557143], ['banana', 'eel', 10.896701], ['jackfruit', 'electric ray', 8.664799], ['custard apple', 'artichoke', 9.747278], ['pomegranate', 'fig', 9.397093], ['hay', 'barrel', 7.6304564], ['carbonara', 'tray', 6.3087525], ['chocolate sauce', 'ice cream', 14.424908], ['dough', 'ice cream', 13.151489], ['meat loaf', 'guacamole', 10.06844], ['pizza', 'potpie', 8.038763], ['potpie', 'tray', 8.084702], ['burrito', 'English setter', 11.469815], ['red wine', 'wine bottle', 20.190735], ['espresso', 'cup', 10.433425], ['cup', 'pitcher', 14.85352], ['eggnog', 'beer glass', 11.557417], ['alp', 'valley', 9.517034], ['bubble', 'water bottle', 10.930498], ['cliff', 'cliff dwelling', 10.962099], ['coral reef', 'goldfish', 9.597891], ['geyser', 'fountain', 12.208312], ['lakeside', 'boathouse', 7.4420133], ['promontory', 'seashore', 11.702433], ['sandbar', 'seashore', 9.537573], ['seashore', 'sandbar', 13.239167], ['valley', 'alp', 11.062243], ['volcano', 'wing', 10.225628], ['ballplayer', 'baseball', 9.770364], ['groom', 'gown', 11.865065], ['scuba diver', 'snorkel', 17.54383], ['rapeseed', 'jigsaw puzzle', 7.5285892], ['daisy', 'sea anemone', 8.684108], ["yellow lady's slipper", 'shower curtain', 7.1650395], ['corn', 'ear', 32.009247], ['acorn', 'fig', 8.418265], ['hip', 'fig', 7.1119504], ['buckeye', 'feather boa', 21.106352], ['coral fungus', 'coral reef', 7.0679526], ['agaric', 'mushroom', 16.159185], ['gyromitra', 'hen-of-the-woods', 5.9961863], ['stinkhorn', 'matchstick', 7.695809], ['earthstar', 'teddy', 6.951969], ['hen-of-the-woods', 'mushroom', 10.577764], ['bolete', 'mushroom', 16.388163], ['ear', 'corn', 28.47574], ['toilet tissue', 'paper towel', 32.459312]]
[['tench', 'coho', 31.077907985353416], ['goldfish', 'tench', 41.12635647933847], ['great white shark', 'tiger shark', 27.784559569809293], ['tiger shark', 'great white shark', 27.784559569809293], ['hammerhead', 'great white shark', 56.52978378764706], ['electric ray', 'stingray', 45.090923767982396], ['stingray', 'electric ray', 45.090923767982396], ['cock', 'hen', 42.421079354877484], ['hen', 'cock', 42.421079354877484], ['ostrich', 'bustard', 42.43417035216504], ['brambling', 'junco', 37.09456776952405], ['goldfinch', 'robin', 33.45461375715799], ['house finch', 'goldfinch', 38.94447638896866], ['junco', 'brambling', 37.09456776952405], ['indigo bunting', 'jay', 36.10726039080904], ['robin', 'goldfinch', 33.45461375715799], ['bulbul', 'quail', 50.70244847391495], ['jay', 'indigo bunting', 36.10726039080904], ['magpie', 'black grouse', 39.67166576352857], ['chickadee', 'apron', 33.68974112733077], ['water ouzel', 'ruddy turnstone', 41.989724491720885], ['kite', 'bald eagle', 47.53460139811972], ['bald eagle', 'vulture', 39.361369168418165], ['vulture', 'bald eagle', 39.361369168418165], ['great grey owl', 'kite', 53.71740472428373], ['European fire salamander', 'spotted salamander', 35.65850090123585], ['common newt', 'eft', 29.599969647366194], ['eft', 'common newt', 29.599969647366194], ['spotted salamander', 'European fire salamander', 35.65850090123585], ['axolotl', 'eft', 55.43028595959523], ['bullfrog', 'tailed frog', 28.11929087020099], ['tree frog', 'tailed frog', 35.34869377172497], ['tailed frog', 'bullfrog', 28.11929087020099], ['loggerhead', 'leatherback turtle', 36.074088259441844], ['leatherback turtle', 'loggerhead', 36.074088259441844], ['mud turtle', 'terrapin', 41.595079453772236], ['terrapin', 'loggerhead', 41.49478594645582], ['box turtle', 'terrapin', 47.72255633149276], ['banded gecko', 'whiptail', 36.324962900015464], ['common iguana', 'African chameleon', 44.75020731446392], ['American chameleon', 'green lizard', 32.282210476998785], ['whiptail', 'alligator lizard', 31.90602427074901], ['agama', 'whiptail', 35.29439524821895], ['frilled lizard', 'common iguana', 50.67948697836852], ['alligator lizard', 'whiptail', 31.90602427074901], ['Gila monster', 'king snake', 58.57862721510509], ['green lizard', 'American chameleon', 32.282210476998785], ['African chameleon', 'green lizard', 42.69428392516746], ['Komodo dragon', 'common iguana', 52.59390318605142], ['African crocodile', 'American alligator', 33.422912267681355], ['American alligator', 'African crocodile', 33.422912267681355], ['triceratops', 'warthog', 60.233331297702186], ['thunder snake', 'garter snake', 43.772177693588226], ['ringneck snake', 'garter snake', 40.965506581661934], ['hognose snake', 'night snake', 35.48195995744704], ['green snake', 'vine snake', 39.94701691185776], ['king snake', 'night snake', 40.60250278519632], ['garter snake', 'ringneck snake', 40.965506581661934], ['water snake', 'sea snake', 39.59605761083182], ['vine snake', 'green snake', 39.94701691185776], ['night snake', 'hognose snake', 35.48195995744704], ['boa constrictor', 'rock python', 41.95333449668363], ['rock python', 'boa constrictor', 41.95333449668363], ['Indian cobra', 'king snake', 57.77431171160761], ['green mamba', 'motor scooter', 55.006085423285214], ['sea snake', 'night snake', 37.538404813437396], ['horned viper', 'sidewinder', 25.95679111297875], ['diamondback', 'hognose snake', 37.17671850012451], ['sidewinder', 'horned viper', 25.95679111297875], ['trilobite', 'chiton', 43.33912111413097], ['harvestman', 'barn spider', 46.2762726265105], ['scorpion', 'limousine', 66.32090103476052], ['black and gold garden spider', 'garden spider', 25.18439564849011], ['barn spider', 'garden spider', 33.92736237859864], ['garden spider', 'black and gold garden spider', 25.18439564849011], ['black widow', 'barn spider', 37.50256664393535], ['tarantula', 'wolf spider', 46.92555086750448], ['wolf spider', 'tarantula', 46.92555086750448], ['tick', 'leaf beetle', 52.48781713184141], ['centipede', 'European fire salamander', 49.73581082988645], ['black grouse', 'prairie chicken', 39.461748748095445], ['ptarmigan', 'quail', 46.34312836737503], ['ruffed grouse', 'partridge', 26.640888470104795], ['prairie chicken', 'black grouse', 39.461748748095445], ['peacock', 'indigo bunting', 55.17574460728872], ['quail', 'ruffed grouse', 39.81926944018436], ['partridge', 'ruffed grouse', 26.640888470104795], ['African grey', 'Bedlington terrier', 67.05557178215133], ['macaw', 'bee eater', 43.02533773836063], ['sulphur-crested cockatoo', 'macaw', 46.499133686486566], ['lorikeet', 'macaw', 44.7305681885302], ['coucal', 'bulbul', 56.4121036609443], ['bee eater', 'jay', 42.30068151046742], ['hornbill', 'toucan', 49.50970795349762], ['hummingbird', 'jacamar', 51.029123405628624], ['jacamar', 'goldfinch', 43.825047240419494], ['toucan', 'hornbill', 49.50970795349762], ['drake', 'red-breasted merganser', 44.42661912980424], ['red-breasted merganser', 'drake', 44.42661912980424], ['goose', 'oystercatcher', 40.024254121329356], ['black swan', 'drake', 47.33746606398969], ['tusker', 'African elephant', 31.64977973270756], ['echidna', 'porcupine', 41.17076166663592], ['platypus', 'beaver', 50.50464087586407], ['wallaby', 'wood rabbit', 47.50904519141592], ['koala', 'lesser panda', 63.352889618760685], ['wombat', 'badger', 55.00643161968653], ['jellyfish', 'sea anemone', 42.5920719703393], ['sea anemone', 'brain coral', 31.897266464544515], ['brain coral', 'sea anemone', 31.897266464544515], ['flatworm', 'sea slug', 36.00723871993484], ['nematode', 'sidewinder', 36.09805507334329], ['conch', 'Eskimo dog', 74.03398712753487], ['snail', 'slug', 38.84696474529091], ['slug', 'snail', 38.84696474529091], ['sea slug', 'rock beauty', 26.81153821665159], ['chiton', 'trilobite', 43.33912111413097], ['chambered nautilus', 'coil', 41.08080386614973], ['Dungeness crab', 'rock crab', 32.11832866344095], ['rock crab', 'Dungeness crab', 32.11832866344095], ['fiddler crab', 'rock crab', 36.591373148150474], ['king crab', 'Dungeness crab', 34.31854739514043], ['American lobster', 'crayfish', 28.135078489344266], ['spiny lobster', 'crayfish', 40.738577710410894], ['crayfish', 'American lobster', 28.135078489344266], ['hermit crab', 'fiddler crab', 49.66411350182217], ['isopod', 'centipede', 60.00597097568187], ['white stork', 'black stork', 29.497467449554037], ['black stork', 'white stork', 29.497467449554037], ['spoonbill', 'American egret', 49.15964208734895], ['flamingo', 'goldfish', 44.20638630157129], ['little blue heron', 'American egret', 39.297312994577], ['American egret', 'white stork', 39.057052745190916], ['bittern', 'red-breasted merganser', 52.58121044067215], ['crane', 'American egret', 42.961705388436485], ['limpkin', 'black stork', 40.017911334039695], ['European gallinule', 'bee eater', 42.95770028378934], ['American coot', 'oystercatcher', 35.37466009940502], ['bustard', 'impala', 41.31737070616077], ['ruddy turnstone', 'red-backed sandpiper', 28.474321542482578], ['red-backed sandpiper', 'ruddy turnstone', 28.474321542482578], ['redshank', 'dowitcher', 33.291542202585966], ['dowitcher', 'red-backed sandpiper', 30.89159117419233], ['oystercatcher', 'American coot', 35.37466009940502], ['pelican', 'white stork', 43.54018466502023], ['king penguin', 'goose', 46.54252402877126], ['albatross', 'bald eagle', 44.255660390084564], ['grey whale', 'great white shark', 53.25762856154975], ['killer whale', 'magpie', 44.80734312639922], ['dugong', 'ice bear', 53.45668971963425], ['sea lion', 'platypus', 64.0781230949768], ['Chihuahua', 'Pembroke', 50.40352548045995], ['Japanese spaniel', 'papillon', 37.74456707593823], ['Maltese dog', 'Lhasa', 40.960993378037415], ['Pekinese', 'Shih-Tzu', 39.90404904120398], ['Shih-Tzu', 'Lhasa', 32.29314428742965], ['Blenheim spaniel', 'Welsh springer spaniel', 40.880103489694406], ['papillon', 'Border collie', 33.296667863621316], ['toy terrier', 'English foxhound', 46.048160717543865], ['Rhodesian ridgeback', 'redbone', 42.89157131242032], ['Afghan hound', 'Saluki', 61.34930565774563], ['basset', 'bloodhound', 40.09538949541534], ['beagle', 'Walker hound', 42.89363178212982], ['bloodhound', 'basset', 40.09538949541534], ['bluetick', 'English setter', 49.56247537042263], ['black-and-tan coonhound', 'bloodhound', 41.311007797370124], ['Walker hound', 'English foxhound', 33.17332486233291], ['English foxhound', 'Walker hound', 33.17332486233291], ['redbone', 'Rhodesian ridgeback', 42.89157131242032], ['borzoi', 'whippet', 45.8271367733546], ['Irish wolfhound', 'Scottish deerhound', 43.4347290033419], ['Italian greyhound', 'whippet', 52.440933569797124], ['whippet', 'borzoi', 45.8271367733546], ['Ibizan hound', 'basenji', 47.70199648599626], ['Norwegian elkhound', 'malamute', 52.4032497650432], ['otterhound', 'bloodhound', 52.9020182522014], ['Saluki', 'whippet', 50.26745120244934], ['Scottish deerhound', 'Irish wolfhound', 43.4347290033419], ['Weimaraner', 'Chesapeake Bay retriever', 54.39379614067904], ['Staffordshire bullterrier', 'American Staffordshire terrier', 30.144071375023316], ['American Staffordshire terrier', 'Staffordshire bullterrier', 30.144071375023316], ['Bedlington terrier', 'komondor', 60.01705893561638], ['Border terrier', 'Norfolk terrier', 43.81854424026359], ['Kerry blue terrier', 'Airedale', 55.896672425836314], ['Irish terrier', 'Airedale', 52.99833705380528], ['Norfolk terrier', 'Lakeland terrier', 38.89389864184999], ['Norwich terrier', 'Australian terrier', 33.858448447174894], ['Yorkshire terrier', 'silky terrier', 45.91456697974511], ['wire-haired fox terrier', 'English foxhound', 47.77993171824338], ['Lakeland terrier', 'Norfolk terrier', 38.89389864184999], ['Sealyham terrier', 'Dandie Dinmont', 49.15529639888768], ['Airedale', 'Lakeland terrier', 46.702353937261556], ['cairn', 'Norwich terrier', 47.42152749048949], ['Australian terrier', 'Norwich terrier', 33.858448447174894], ['Dandie Dinmont', 'Lakeland terrier', 44.48590893474022], ['Boston bull', 'Appenzeller', 41.42284897242327], ['miniature schnauzer', 'standard schnauzer', 38.56585530378524], ['giant schnauzer', 'standard schnauzer', 48.45137905996588], ['standard schnauzer', 'miniature schnauzer', 38.56585530378524], ['Scotch terrier', 'schipperke', 45.70700036988317], ['Tibetan terrier', 'green mamba', 59.86475235243816], ['silky terrier', 'Yorkshire terrier', 45.91456697974511], ['soft-coated wheaten terrier', 'Lakeland terrier', 46.15279586770719], ['West Highland white terrier', 'cairn', 55.02501650527921], ['Lhasa', 'Shih-Tzu', 32.29314428742965], ['flat-coated retriever', 'magpie', 47.312845730242756], ['curly-coated retriever', 'Irish water spaniel', 43.880027200417445], ['golden retriever', 'Labrador retriever', 37.98480241943809], ['Labrador retriever', 'golden retriever', 37.98480241943809], ['Chesapeake Bay retriever', 'Rhodesian ridgeback', 45.1475094265661], ['German short-haired pointer', 'Sussex spaniel', 57.396973767405626], ['vizsla', 'Rhodesian ridgeback', 44.8248599150767], ['English setter', 'Welsh springer spaniel', 49.296040504030344], ['Irish setter', 'redbone', 49.87847536450468], ['Gordon setter', 'black-and-tan coonhound', 49.764118894007105], ['Brittany spaniel', 'Welsh springer spaniel', 41.529424691829945], ['clumber', 'Welsh springer spaniel', 55.7974175624699], ['English springer', 'Welsh springer spaniel', 35.74671504794182], ['Welsh springer spaniel', 'English springer', 35.74671504794182], ['cocker spaniel', 'Sussex spaniel', 54.832907136847346], ['Sussex spaniel', 'Tibetan mastiff', 53.2797676663302], ['Irish water spaniel', 'curly-coated retriever', 43.880027200417445], ['kuvasz', 'Great Pyrenees', 42.19386526054469], ['schipperke', 'Scotch terrier', 45.70700036988317], ['groenendael', 'schipperke', 45.96070842306502], ['malinois', 'German shepherd', 38.751860149405246], ['briard', 'soft-coated wheaten terrier', 53.57841207272174], ['kelpie', 'schipperke', 49.41867876164715], ['komondor', 'wool', 55.23980922701942], ['Old English sheepdog', 'Dandie Dinmont', 55.39052142091867], ['Shetland sheepdog', 'collie', 49.14007104218766], ['collie', 'Shetland sheepdog', 49.14007104218766], ['Border collie', 'papillon', 33.296667863621316], ['Bouvier des Flandres', 'briard', 56.6822325626051], ['Rottweiler', 'EntleBucher', 50.6803950399585], ['German shepherd', 'malinois', 38.751860149405246], ['Doberman', 'Rottweiler', 52.6921775614927], ['miniature pinscher', 'Doberman', 54.5360715991604], ['Greater Swiss Mountain dog', 'EntleBucher', 20.730237259047623], ['Bernese mountain dog', 'Appenzeller', 41.966572856318486], ['Appenzeller', 'Greater Swiss Mountain dog', 28.759391511255174], ['EntleBucher', 'Greater Swiss Mountain dog', 20.730237259047623], ['boxer', 'Staffordshire bullterrier', 48.56598885472734], ['bull mastiff', 'Staffordshire bullterrier', 53.26941996601451], ['Tibetan mastiff', 'Sussex spaniel', 53.2797676663302], ['French bulldog', 'kelpie', 52.03423180269768], ['Great Dane', 'Rhodesian ridgeback', 48.886192971853006], ['Saint Bernard', 'Welsh springer spaniel', 46.279095050269454], ['Eskimo dog', 'Siberian husky', 37.86826659928303], ['malamute', 'Siberian husky', 47.28153915972385], ['Siberian husky', 'Eskimo dog', 37.86826659928303], ['dalmatian', 'English setter', 52.78941803287852], ['affenpinscher', 'Lhasa', 50.18600655467866], ['basenji', 'Ibizan hound', 47.70199648599626], ['pug', 'boxer', 55.20622735763149], ['Leonberg', 'Saint Bernard', 54.37764226697977], ['Newfoundland', 'Tibetan mastiff', 54.52472203046981], ['Great Pyrenees', 'kuvasz', 42.19386526054469], ['Samoyed', 'white wolf', 50.30957386677262], ['Pomeranian', 'Persian cat', 54.420015490052], ['chow', 'Pomeranian', 59.31213571740083], ['keeshond', 'malamute', 56.1848907034567], ['Brabancon griffon', 'Border terrier', 47.62204715255319], ['Pembroke', 'Cardigan', 36.858667927194375], ['Cardigan', 'Pembroke', 36.858667927194375], ['toy poodle', 'miniature poodle', 30.185389095450635], ['miniature poodle', 'toy poodle', 30.185389095450635], ['standard poodle', 'miniature poodle', 43.13762835934249], ['Mexican hairless', 'kelpie', 73.83180711717343], ['timber wolf', 'white wolf', 51.172468985102235], ['white wolf', 'Eskimo dog', 45.430793865504924], ['red wolf', 'mashed potato', 58.77855439725657], ['coyote', 'red fox', 45.98715694099413], ['dingo', 'dhole', 53.542258427339426], ['dhole', 'red fox', 44.13128496316417], ['African hunting dog', 'hyena', 44.456543695430824], ['hyena', 'African hunting dog', 44.456543695430824], ['red fox', 'kit fox', 22.44502289449434], ['kit fox', 'red fox', 22.44502289449434], ['Arctic fox', 'white wolf', 49.67860081469938], ['grey fox', 'kit fox', 49.9781202127491], ['tabby', 'tiger cat', 39.095447043002174], ['tiger cat', 'tiger', 35.2639980794418], ['Persian cat', 'Pomeranian', 54.420015490052], ['Siamese cat', 'schipperke', 66.4917207310899], ['Egyptian cat', 'tabby', 40.4714918231417], ['cougar', 'motor scooter', 56.67440802858575], ['lynx', 'leopard', 59.331544541958245], ['leopard', 'jaguar', 40.23641378329616], ['snow leopard', 'jaguar', 48.90724090449823], ['jaguar', 'leopard', 40.23641378329616], ['lion', 'cheetah', 52.58792859939437], ['tiger', 'tiger cat', 35.2639980794418], ['cheetah', 'leopard', 46.85776506070659], ['brown bear', 'American black bear', 51.04253692923065], ['American black bear', 'brown bear', 51.04253692923065], ['ice bear', 'dugong', 53.45668971963425], ['sloth bear', 'giant panda', 59.23034805620658], ['mongoose', 'mink', 56.76870432743401], ['meerkat', 'marmot', 54.24843837758143], ['tiger beetle', 'ground beetle', 43.748055202310134], ['ladybug', 'bee', 37.1809835414473], ['ground beetle', 'long-horned beetle', 40.4645087063188], ['long-horned beetle', 'ground beetle', 40.4645087063188], ['leaf beetle', 'ladybug', 38.19192384359552], ['dung beetle', 'rhinoceros beetle', 49.29917783844752], ['rhinoceros beetle', 'dung beetle', 49.29917783844752], ['weevil', 'ground beetle', 47.026333837741916], ['fly', 'cicada', 47.79615989167566], ['bee', 'ladybug', 37.1809835414473], ['ant', 'candle', 14.353483599986385], ['grasshopper', 'mantis', 27.251073973604118], ['cricket', 'grasshopper', 28.2332162877899], ['walking stick', 'mantis', 43.31668788375619], ['cockroach', 'ground beetle', 51.45392925654463], ['mantis', 'grasshopper', 27.251073973604118], ['cicada', 'leafhopper', 43.56874002564081], ['leafhopper', 'cicada', 43.56874002564081], ['lacewing', 'cicada', 53.375420812158474], ['dragonfly', 'damselfly', 37.323153969963805], ['damselfly', 'dragonfly', 37.323153969963805], ['admiral', 'monarch', 45.17648227541903], ['ringlet', 'lycaenid', 49.34532335996999], ['monarch', 'rapeseed', 39.90250110865937], ['cabbage butterfly', 'admiral', 47.22593374788237], ['sulphur butterfly', 'cabbage butterfly', 48.96959505891768], ['lycaenid', 'ringlet', 49.34532335996999], ['starfish', 'sea cucumber', 61.4727693144493], ['sea urchin', 'lionfish', 39.716988921422214], ['sea cucumber', 'buckeye', 54.185418428877156], ['wood rabbit', 'hare', 30.53030540855829], ['hare', 'wood rabbit', 30.53030540855829], ['Angora', 'green mamba', 55.244076234329185], ['hamster', 'guinea pig', 48.21024562479562], ['porcupine', 'echidna', 41.17076166663592], ['fox squirrel', 'hamster', 64.46631115115476], ['marmot', 'meerkat', 54.24843837758143], ['beaver', 'platypus', 50.50464087586407], ['guinea pig', 'hamster', 48.21024562479562], ['sorrel', 'oxcart', 48.704516041251246], ['zebra', 'impala', 52.551428029032905], ['hog', 'guinea pig', 54.41045001020024], ['wild boar', 'warthog', 40.129626996654544], ['warthog', 'wild boar', 40.129626996654544], ['hippopotamus', 'chimpanzee', 61.8319768354126], ['ox', 'oxcart', 32.95521419408657], ['water buffalo', 'bison', 51.87034758881393], ['bison', 'water buffalo', 51.87034758881393], ['ram', 'bighorn', 40.89561759022521], ['bighorn', 'ram', 40.89561759022521], ['ibex', 'hartebeest', 50.698805671182974], ['hartebeest', 'impala', 40.91800585469678], ['impala', 'gazelle', 27.284292499302598], ['gazelle', 'impala', 27.284292499302598], ['Arabian camel', 'sorrel', 57.227743695147545], ['llama', 'German short-haired pointer', 67.09027121861634], ['weasel', 'black-footed ferret', 42.977226436654796], ['mink', 'otter', 48.53912222855652], ['polecat', 'black-footed ferret', 27.7544997407515], ['black-footed ferret', 'polecat', 27.7544997407515], ['otter', 'mink', 48.53912222855652], ['skunk', 'magpie', 53.38929660697451], ['badger', 'warthog', 51.934350461062856], ['armadillo', 'grey fox', 77.01016445695659], ['three-toed sloth', 'orangutan', 68.78104057782748], ['orangutan', 'gorilla', 45.47700741171301], ['gorilla', 'orangutan', 45.47700741171301], ['chimpanzee', 'gorilla', 46.25418055430233], ['gibbon', 'spider monkey', 42.973673013434926], ['siamang', 'German short-haired pointer', 57.84818155633762], ['guenon', 'patas', 49.81047331589136], ['patas', 'langur', 43.88853206784561], ['baboon', 'langur', 44.00284497860266], ['macaque', 'baboon', 47.877465565591244], ['langur', 'patas', 43.88853206784561], ['colobus', 'gibbon', 51.98190697101973], ['proboscis monkey', 'patas', 45.04288998806096], ['marmoset', 'titi', 40.74423158223213], ['capuchin', 'howler monkey', 48.58452233527541], ['howler monkey', 'capuchin', 48.58452233527541], ['titi', 'marmoset', 40.74423158223213], ['spider monkey', 'gibbon', 42.973673013434926], ['squirrel monkey', 'capuchin', 50.22497725080371], ['Madagascar cat', 'indri', 42.958325440004934], ['indri', 'Madagascar cat', 42.958325440004934], ['Indian elephant', 'tusker', 32.50866583685322], ['African elephant', 'tusker', 31.64977973270756], ['lesser panda', 'orangutan', 51.80516053640916], ['giant panda', 'lesser panda', 51.82964178516117], ['barracouta', 'coho', 35.277343533731035], ['eel', 'barracouta', 57.7993735097968], ['coho', 'tench', 31.077907985353416], ['rock beauty', 'sea slug', 26.81153821665159], ['anemone fish', 'sea slug', 34.20494156322665], ['sturgeon', 'barracouta', 42.409914289917225], ['gar', 'barracouta', 41.23685574197401], ['lionfish', 'sea urchin', 39.716988921422214], ['puffer', 'lionfish', 59.86252969042906], ['abacus', 'toyshop', 50.48051817130793], ['abaya', 'cloak', 36.11459084462954], ['academic gown', 'mortarboard', 18.943408409469736], ['accordion', 'library', 51.86271032886972], ['acoustic guitar', 'violin', 33.53685771200564], ['aircraft carrier', 'tank', 46.660938310995206], ['airliner', 'warplane', 30.145205231305027], ['airship', 'balloon', 60.309476122382705], ['altar', 'church', 24.595382927048014], ['ambulance', 'police van', 43.61623803932573], ['amphibian', 'harvester', 44.355095668014854], ['analog clock', 'wall clock', 16.509403208084457], ['apiary', 'crate', 44.82250830750732], ['apron', 'chickadee', 33.68974112733077], ['ashcan', 'cocktail shaker', 44.02730476026355], ['assault rifle', 'rifle', 23.126126978450962], ['backpack', 'tray', 15.724673783598371], ['bakery', 'confectionery', 45.368903473014974], ['balance beam', 'horizontal bar', 37.42306235068323], ['balloon', 'parachute', 45.21502446643151], ['ballpoint', 'fountain pen', 49.71620878562267], ['Band Aid', 'bib', 55.54913629363421], ['banjo', 'electric guitar', 53.16080594820304], ['bannister', 'prison', 37.72374009037611], ['barbell', 'dumbbell', 39.29032313671523], ['barber chair', 'rocking chair', 52.09363283580105], ['barbershop', 'vestment', 52.965025680277925], ['barn', 'thatch', 46.81160296203282], ['barometer', 'stopwatch', 37.5490304468704], ['barrel', 'rain barrel', 19.380336494162147], ['barrow', 'plow', 60.907396410452485], ['baseball', 'golf ball', 55.38322411770441], ['basketball', 'volleyball', 41.629351780774826], ['bassinet', 'cradle', 29.89777301162559], ['bassoon', 'oboe', 47.304168169114604], ['bathing cap', 'bikini', 52.65981766474676], ['bath towel', 'sandal', 47.47644358323926], ['bathtub', 'tub', 16.951423805774215], ['beach wagon', 'cab', 29.81094708186858], ['beacon', 'breakwater', 49.32497465787489], ['beaker', 'measuring cup', 30.831811265923058], ['bearskin', 'pickelhaube', 40.926583408491204], ['beer bottle', 'pop bottle', 38.944760057516085], ['beer glass', 'red wine', 46.55854639255289], ['bell cote', 'church', 27.17297115385296], ['bib', 'purse', 33.15653164642394], ['bicycle-built-for-two', 'unicycle', 47.237274555396134], ['bikini', 'maillot', 22.796319443624693], ['binder', 'wallet', 37.541620783013954], ['binoculars', 'reflex camera', 55.611027915164904], ['birdhouse', 'barn', 53.0931338251473], ['boathouse', 'lakeside', 45.366656752750416], ['bobsled', 'bullet train', 52.79700216394866], ['bolo tie', 'stethoscope', 49.47800519470116], ['bonnet', 'diaper', 54.95825830801045], ['bookcase', 'library', 32.88154194652754], ['bookshop', 'library', 20.506126418783364], ['bottlecap', 'necklace', 41.154787725541425], ['bow', 'horizontal bar', 59.96950315446802], ['bow tie', 'monitor', 16.624365772970577], ['brass', 'menu', 41.71193910929834], ['brassiere', 'honeycomb', 52.502864505186764], ['breakwater', 'lakeside', 34.95544038486713], ['breastplate', 'cuirass', 25.247710775713163], ['broom', 'swab', 51.78987509262187], ['bucket', 'German short-haired pointer', 63.70061851213161], ['buckle', 'wallet', 47.06634902918963], ['bulletproof vest', 'military uniform', 46.546379340261254], ['bullet train', 'trailer truck', 46.135563661615755], ['butcher shop', 'sliding door', 48.65889925350244], ['cab', 'beach wagon', 29.81094708186858], ['caldron', 'pot', 54.15803516965419], ['candle', 'carton', 13.19336616973363], ['cannon', 'tank', 58.971441048824644], ['canoe', 'paddle', 44.02690134443017], ['can opener', "carpenter's kit", 44.748240516213315], ['cardigan', 'stole', 38.252125400508795], ['car mirror', 'car wheel', 42.98578071583672], ['carousel', 'sliding door', 40.19720825825859], ["carpenter's kit", 'can opener', 44.748240516213315], ['carton', 'candle', 13.19336616973363], ['car wheel', 'disk brake', 39.746821438363156], ['cash machine', 'slot', 39.580333288176405], ['cassette', 'tape player', 35.884230220469064], ['cassette player', 'tape player', 20.059734403424624], ['castle', 'cliff dwelling', 61.8531606028544], ['catamaran', 'trimaran', 22.3432651280432], ['CD player', 'cassette player', 39.77597636637519], ['cello', 'violin', 31.501572342640195], ['cellular telephone', 'hand-held computer', 30.618000834438167], ['chain', 'hook', 45.95688366556201], ['chainlink fence', 'worm fence', 36.56072044280412], ['chain mail', 'cuirass', 54.625531855248326], ['chain saw', 'padlock', 49.347797094956526], ['chest', 'mailbox', 38.914339332231535], ['chiffonier', 'wardrobe', 32.000944123645596], ['chime', 'altar', 52.13813627189075], ['china cabinet', 'sliding door', 49.168014525133614], ['Christmas stocking', 'sock', 46.623931790712376], ['church', 'altar', 24.595382927048014], ['cinema', 'breakwater', 49.301136404562214], ['cleaver', 'hatchet', 46.56382128997791], ['cliff dwelling', 'prison', 35.01474647492289], ['cloak', 'abaya', 36.11459084462954], ['clog', 'running shoe', 38.76722462439277], ['cocktail shaker', 'pot', 38.66582023308021], ['coffee mug', 'cup', 34.77271521800983], ['coffeepot', 'teapot', 38.495514843757284], ['coil', 'chambered nautilus', 41.08080386614973], ['combination lock', 'vending machine', 50.312626164438086], ['computer keyboard', 'space bar', 15.71922177162307], ['confectionery', 'grocery store', 31.13442102965081], ['container ship', 'dock', 38.030808658593656], ['convertible', 'sports car', 29.280155645905257], ['corkscrew', 'Madagascar cat', 49.57636450939853], ['cornet', 'trombone', 32.82054216921546], ['cowboy boot', 'shoe shop', 43.43618478224693], ['cowboy hat', 'sombrero', 22.93400725919576], ['cradle', 'crib', 26.679001180777107], ['crane', 'tow truck', 41.14048554381985], ['crash helmet', 'car wheel', 52.2853545534622], ['crate', 'apiary', 44.82250830750732], ['crib', 'cradle', 26.679001180777107], ['Crock Pot', 'washer', 51.059638364581815], ['croquet ball', 'hip', 52.83964492851816], ['crutch', 'spindle', 79.64150633463684], ['cuirass', 'breastplate', 25.247710775713163], ['dam', 'lakeside', 49.74985523298661], ['desk', 'candle', 15.667036228014597], ['desktop computer', 'notebook', 28.498454804945677], ['dial telephone', 'car wheel', 46.329366024693236], ['diaper', 'bib', 38.38341278408064], ['digital clock', 'digital watch', 32.65094155418799], ['digital watch', 'digital clock', 32.65094155418799], ['dining table', 'restaurant', 28.523042699609555], ['dishrag', 'prayer rug', 42.99415726111805], ['dishwasher', 'refrigerator', 56.98262481888668], ['disk brake', 'car wheel', 39.746821438363156], ['dock', 'lakeside', 29.90675919596553], ['dogsled', 'Eskimo dog', 39.862792630824615], ['dome', 'planetarium', 47.553002911882444], ['doormat', 'street sign', 42.4158447026101], ['drilling platform', 'dock', 39.74174442572948], ['drum', 'potpie', 70.10833204873904], ['drumstick', 'matchstick', 66.65780662153345], ['dumbbell', 'barbell', 39.29032313671523], ['Dutch oven', 'frying pan', 25.342046075966444], ['electric fan', 'car wheel', 56.86084988812601], ['electric guitar', 'acoustic guitar', 35.37251882310086], ['electric locomotive', 'passenger car', 31.15896897614313], ['entertainment center', 'home theater', 28.721961106524176], ['envelope', 'handkerchief', 41.057664467471085], ['espresso maker', 'coffeepot', 55.28926065476007], ['face powder', 'perfume', 43.95457888414826], ['feather boa', 'buckeye', 57.01087821087305], ['file', 'chiffonier', 43.260261156314115], ['fireboat', 'fountain', 53.37621439997291], ['fire engine', 'electric locomotive', 39.20043926382155], ['fire screen', 'wardrobe', 48.13573338098216], ['flagpole', 'schooner', 45.09596700631194], ['flute', 'oboe', 49.59941829217229], ['folding chair', 'patio', 36.29797135666641], ['football helmet', 'crash helmet', 65.67254973778999], ['forklift', 'tow truck', 43.59149859777707], ['fountain', 'fireboat', 53.37621439997291], ['fountain pen', 'ballpoint', 49.71620878562267], ['four-poster', 'carousel', 47.43765954353935], ['freight car', 'moving van', 42.66602166006194], ['French horn', 'trombone', 42.65907442572301], ['frying pan', 'Dutch oven', 25.342046075966444], ['fur coat', 'cloak', 50.978558988681705], ['garbage truck', 'trailer truck', 34.003825252990836], ['gasmask', 'scuba diver', 63.46067926283172], ['gas pump', 'street sign', 48.821244515227434], ['goblet', 'beer glass', 61.55986669494988], ['go-kart', 'lawn mower', 41.73391285779318], ['golf ball', 'car wheel', 48.08439084129199], ['golfcart', 'Model T', 41.94358451809481], ['gondola', 'lakeside', 40.15566177827932], ['gong', 'shield', 61.010110986443266], ['gown', 'overskirt', 25.89488215603616], ['grand piano', 'upright', 30.031540630097343], ['greenhouse', 'sliding door', 46.74733134930003], ['grille', 'car wheel', 43.42822517887992], ['grocery store', 'confectionery', 31.13442102965081], ['guillotine', 'cassette', 59.62726427173437], ['hair slide', 'necklace', 36.29506728440777], ['hair spray', 'lotion', 61.610912515686906], ['half track', 'tank', 34.35985284172049], ['hammer', 'hatchet', 50.927788568103956], ['hamper', 'ashcan', 45.31189654770588], ['hand blower', 'dumbbell', 61.285542541527484], ['hand-held computer', 'cellular telephone', 30.618000834438167], ['handkerchief', 'envelope', 41.057664467471085], ['hard disc', 'EntleBucher', 55.435205526080175], ['harmonica', 'brass', 52.65476166294222], ['harp', 'sidewinder', 75.11592278251496], ['harvester', 'tractor', 29.482276894414145], ['hatchet', 'cleaver', 46.56382128997791], ['holster', 'scabbard', 41.86563269761816], ['home theater', 'entertainment center', 28.721961106524176], ['honeycomb', 'apiary', 46.738026437092685], ['hook', 'chain', 45.95688366556201], ['hoopskirt', 'overskirt', 21.486594345597574], ['horizontal bar', 'parallel bars', 33.7648874861612], ['horse cart', 'oxcart', 46.69325704966671], ['hourglass', 'organ', 63.66653543561955], ['iPod', 'cellular telephone', 46.73637575122161], ['iron', 'lycaenid', 68.17187320937425], ["jack-o'-lantern", 'comic book', 49.405943626501056], ['jean', 'swimming trunks', 35.55399997275271], ['jeep', 'beach wagon', 36.70204739797393], ['jersey', 'sweatshirt', 24.6938962738486], ['jigsaw puzzle', 'sidewinder', 41.683271430964965], ['jinrikisha', 'horse cart', 47.92113883115676], ['joystick', 'dining table', 53.32295838215468], ['kimono', 'vestment', 57.07970544280603], ['knee pad', 'sock', 50.946890909646044], ['knot', 'chain', 49.339140953994374], ['lab coat', 'Windsor tie', 46.69124399311931], ['ladle', 'wooden spoon', 50.45412907409066], ['lampshade', 'table lamp', 30.10256819303639], ['laptop', 'notebook', 14.168177146420204], ['lawn mower', 'go-kart', 41.73391285779318], ['lens cap', 'rugby ball', 44.33724461178914], ['letter opener', "carpenter's kit", 45.36692851764378], ['library', 'bookshop', 20.506126418783364], ['lifeboat', 'speedboat', 38.54391709364624], ['lighter', 'torch', 56.76932576362212], ['limousine', 'banded gecko', 53.11669282476672], ['liner', 'container ship', 44.76112815546543], ['lipstick', 'green mamba', 66.72067839718996], ['Loafer', 'shoe shop', 46.473068477882975], ['lotion', 'sunscreen', 34.64843661922209], ['loudspeaker', 'tape player', 39.47614118244398], ['loupe', 'sunglasses', 45.96243211597516], ['lumbermill', 'harvester', 45.19867168263908], ['magnetic compass', 'scale', 43.04808726741308], ['mailbag', 'purse', 38.34579380249873], ['mailbox', 'chest', 38.914339332231535], ['maillot', 'maillot', 46.32322644844323], ['maillot', 'bikini', 22.796319443624693], ['manhole cover', 'prayer rug', 37.00445364876706], ['maraca', 'plunger', 70.577739303241], ['marimba', 'park bench', 74.56338405804487], ['mask', 'ski mask', 51.00398029428071], ['matchstick', 'spindle', 61.476333654581055], ['maypole', 'schooner', 54.117347088150716], ['maze', 'manhole cover', 45.87965639685797], ['measuring cup', 'beaker', 30.831811265923058], ['medicine chest', 'washbasin', 46.099703005415336], ['megalith', 'stone wall', 43.84417319824266], ['microphone', 'joystick', 75.93436850539089], ['microwave', 'radio', 40.65502195973918], ['military uniform', 'bulletproof vest', 46.546379340261254], ['milk can', 'rain barrel', 47.34253040920896], ['minibus', 'minivan', 29.13325036544101], ['miniskirt', 'swimming trunks', 33.05887784388446], ['minivan', 'minibus', 29.13325036544101], ['missile', 'projectile', 18.38185126843016], ['mitten', 'sock', 51.85868531102095], ['mixing bowl', 'mortar', 45.461412101191655], ['mobile home', 'moving van', 46.80967844467371], ['Model T', 'golfcart', 41.94358451809481], ['modem', 'radio', 42.67141442904137], ['monastery', 'church', 27.958322296671007], ['monitor', 'bow tie', 16.624365772970577], ['moped', 'mountain bike', 51.14947744570564], ['mortar', 'mixing bowl', 45.461412101191655], ['mortarboard', 'academic gown', 18.943408409469736], ['mosque', 'palace', 38.59101078335796], ['mosquito net', 'recreational vehicle', 71.76431354941501], ['motor scooter', 'green mamba', 55.006085423285214], ['mountain bike', 'unicycle', 40.76010586920439], ['mountain tent', 'umbrella', 40.93593597918811], ['mouse', 'modem', 52.02685088385251], ['mousetrap', 'chain saw', 60.19624482162488], ['moving van', 'trailer truck', 24.706998826559754], ['muzzle', 'football helmet', 72.57768601858632], ['nail', 'screw', 47.274655645559704], ['neck brace', 'toilet seat', 87.4609371510719], ['necklace', 'hair slide', 36.29506728440777], ['nipple', 'oxygen mask', 48.566617224115475], ['notebook', 'laptop', 14.168177146420204], ['obelisk', 'brass', 55.31560505055852], ['oboe', 'bassoon', 47.304168169114604], ['ocarina', 'harmonica', 59.861693621891874], ['odometer', 'barometer', 46.18361208687152], ['oil filter', 'car wheel', 46.826042141699844], ['organ', 'altar', 53.683655002739194], ['oscilloscope', 'digital clock', 54.209653415581805], ['overskirt', 'hoopskirt', 21.486594345597574], ['oxcart', 'ox', 32.95521419408657], ['oxygen mask', 'nipple', 48.566617224115475], ['packet', 'sunscreen', 47.84486751634129], ['paddle', 'canoe', 44.02690134443017], ['paddlewheel', 'thresher', 47.64226229147814], ['padlock', 'chain saw', 49.347797094956526], ['paintbrush', 'ballpoint', 83.57282448372497], ['pajama', 'apron', 51.1517040397666], ['palace', 'monastery', 35.64708060232633], ['panpipe', 'comic book', 65.03238751407832], ['paper towel', 'toilet tissue', 24.520790465216706], ['parachute', 'umbrella', 40.21868929676849], ['parallel bars', 'horizontal bar', 33.7648874861612], ['park bench', 'folding chair', 50.85696998921952], ['parking meter', 'digital watch', 63.00280561709137], ['passenger car', 'electric locomotive', 31.15896897614313], ['patio', 'dining table', 30.9791337205258], ['pay-phone', 'cash machine', 62.59398987410353], ['pedestal', 'throne', 56.768196852006184], ['pencil box', 'wallet', 38.60753802287191], ['pencil sharpener', 'Polaroid camera', 54.11960946742179], ['perfume', 'face powder', 43.95457888414826], ['Petri dish', 'perfume', 46.838094847103356], ['photocopier', 'printer', 35.99328041942777], ['pick', 'viaduct', 54.770909605944105], ['pickelhaube', 'bearskin', 40.926583408491204], ['picket fence', 'worm fence', 46.62819400923115], ['pickup', 'beach wagon', 35.21648658986832], ['pier', 'steel arch bridge', 38.06281108760393], ['piggy bank', 'hog', 56.21232896557814], ['pill bottle', 'lotion', 48.0252415727917], ['pillow', 'studio couch', 44.55660325422807], ['ping-pong ball', 'golf ball', 58.588105978170816], ['pinwheel', 'folding chair', 46.471032749075256], ['pirate', 'schooner', 39.52259663194272], ['pitcher', 'water jug', 29.683772381108124], ['plane', "carpenter's kit", 64.71832749741374], ['planetarium', 'dome', 47.553002911882444], ['plastic bag', 'packet', 48.62917217801214], ['plate rack', 'china cabinet', 49.91089227782599], ['plow', 'harvester', 40.36618427604735], ['plunger', 'maraca', 70.577739303241], ['Polaroid camera', 'reflex camera', 44.76211128225871], ['pole', 'totem pole', 46.612185110561484], ['police van', 'cab', 38.01140383212731], ['poncho', 'stole', 31.611694914558026], ['pool table', 'croquet ball', 54.149624973453186], ['pop bottle', 'water bottle', 27.81417875902617], ['pot', 'cocktail shaker', 38.66582023308021], ["potter's wheel", 'coil', 57.45916943659863], ['power drill', 'chain saw', 73.70322934325877], ['prayer rug', 'sidewinder', 35.184224980806384], ['printer', 'photocopier', 35.99328041942777], ['prison', 'cliff dwelling', 35.01474647492289], ['projectile', 'missile', 18.38185126843016], ['projector', 'radio', 43.14989091454925], ['puck', 'ski', 39.530181556707404], ['punching bag', 'chime', 72.22253614047696], ['purse', 'bib', 33.15653164642394], ['quill', 'daisy', 79.72527903486761], ['quilt', 'studio couch', 36.045274875083315], ['racer', 'sports car', 26.598518777779244], ['racket', 'ping-pong ball', 61.499325136032994], ['radiator', 'mailbox', 63.33681588218459], ['radio', 'tape player', 32.19871998013858], ['radio telescope', 'solar dish', 54.433046423737395], ['rain barrel', 'barrel', 19.380336494162147], ['recreational vehicle', 'trailer truck', 52.55285889954632], ['reel', 'lifeboat', 56.33940377718555], ['reflex camera', 'Polaroid camera', 44.76211128225871], ['refrigerator', 'vending machine', 43.09644944862772], ['remote control', 'hand-held computer', 55.7230285268409], ['restaurant', 'dining table', 28.523042699609555], ['revolver', 'holster', 58.45459726493931], ['rifle', 'assault rifle', 23.126126978450962], ['rocking chair', 'folding chair', 45.341084733867476], ['rotisserie', 'microwave', 60.90930437441393], ['rubber eraser', 'backpack', 33.45326731053336], ['rugby ball', 'magpie', 43.8103977629562], ['rule', 'slide rule', 59.20580745770427], ['running shoe', 'clog', 38.76722462439277], ['safe', 'backpack', 29.853921900347217], ['safety pin', 'trench coat', 54.29352507651695], ['saltshaker', 'thimble', 54.107482189677334], ['sandal', 'chickadee', 41.76402915020742], ['sarong', 'jean', 63.69329585992548], ['sax', 'trombone', 47.110336329329044], ['scabbard', 'holster', 41.86563269761816], ['scale', 'magnetic compass', 43.04808726741308], ['school bus', 'street sign', 46.989856680557935], ['schooner', 'trimaran', 39.480105254893886], ['scoreboard', 'menu', 46.133970799387626], ['screen', 'television', 35.03552361588253], ['screw', 'nail', 47.274655645559704], ['screwdriver', 'ballpoint', 53.24026099320889], ['seat belt', 'car mirror', 48.369458258156044], ['sewing machine', 'damselfly', 71.93188306155518], ['shield', 'breastplate', 56.5235946136379], ['shoe shop', 'cowboy boot', 43.43618478224693], ['shoji', 'window shade', 29.342636136007894], ['shopping basket', 'apron', 41.27225666136848], ['shopping cart', 'folding chair', 57.1366320897767], ['shovel', 'paddle', 65.29038202044003], ['shower cap', 'orange', 67.62014047927954], ['shower curtain', 'window shade', 42.628409240951775], ['ski', 'snowmobile', 32.27613327672438], ['ski mask', 'mask', 51.00398029428071], ['sleeping bag', 'diaper', 51.05905262995731], ['slide rule', 'barometer', 49.10715274959316], ['sliding door', 'wardrobe', 27.571849941193417], ['slot', 'cash machine', 39.580333288176405], ['snorkel', 'scuba diver', 47.666791866822756], ['snowmobile', 'ski', 32.27613327672438], ['snowplow', 'harvester', 36.40810081132357], ['soap dispenser', 'medicine chest', 53.399970831130844], ['soccer ball', 'volleyball', 52.22794325493418], ['sock', 'Christmas stocking', 46.623931790712376], ['solar dish', 'radio telescope', 54.433046423737395], ['sombrero', 'cowboy hat', 22.93400725919576], ['soup bowl', 'consomme', 27.428323449421654], ['space bar', 'computer keyboard', 15.71922177162307], ['space heater', 'modem', 51.53895058821411], ['space shuttle', 'missile', 49.07711267262216], ['spatula', 'letter opener', 66.90595807969945], ['speedboat', 'lifeboat', 38.54391709364624], ['spider web', 'black and gold garden spider', 27.11801352818535], ['spindle', 'wool', 60.31755969709401], ['sports car', 'racer', 26.598518777779244], ['spotlight', 'loupe', 49.20263128920292], ['stage', 'home theater', 47.49292453470701], ['steam locomotive', 'electric locomotive', 43.61158349388096], ['steel arch bridge', 'pier', 38.06281108760393], ['steel drum', 'strainer', 54.20863783582488], ['stethoscope', 'bolo tie', 49.47800519470116], ['stole', 'poncho', 31.611694914558026], ['stone wall', 'worm fence', 33.22389423950028], ['stopwatch', 'analog clock', 37.28595815828326], ['stove', 'fire screen', 52.15396564799747], ['strainer', 'Petri dish', 51.21515583789627], ['streetcar', 'trolleybus', 31.213110062057158], ['stretcher', 'jinrikisha', 55.905287075972296], ['studio couch', 'quilt', 36.045274875083315], ['stupa', 'mosque', 62.586892722238574], ['submarine', 'dock', 57.859533232972936], ['suit', 'Windsor tie', 31.61707456892359], ['sundial', 'manhole cover', 57.25677370203132], ['sunglass', 'sunglasses', 19.89558927642632], ['sunglasses', 'sunglass', 19.89558927642632], ['sunscreen', 'lotion', 34.64843661922209], ['suspension bridge', 'pier', 38.91765489215258], ['swab', 'broom', 51.78987509262187], ['sweatshirt', 'jersey', 24.6938962738486], ['swimming trunks', 'miniskirt', 33.05887784388446], ['swing', 'chain', 57.43026936941659], ['switch', 'gas pump', 54.33859683014229], ['syringe', 'ballpoint', 56.28119577662574], ['table lamp', 'lampshade', 30.10256819303639], ['tank', 'half track', 34.35985284172049], ['tape player', 'cassette player', 20.059734403424624], ['teapot', 'pitcher', 35.66787784219332], ['teddy', 'toyshop', 42.759894402768786], ['television', 'screen', 35.03552361588253], ['tennis ball', 'volleyball', 58.23170648903396], ['thatch', 'barn', 46.81160296203282], ['theater curtain', 'prayer rug', 41.46419442055172], ['thimble', 'saltshaker', 54.107482189677334], ['thresher', 'harvester', 31.834579150469384], ['throne', 'altar', 40.002538981919635], ['tile roof', 'sidewinder', 49.30338209827116], ['toaster', 'Crock Pot', 58.72806272611502], ['tobacco shop', 'bookshop', 24.133435923883436], ['toilet seat', 'medicine chest', 53.05184991399334], ['torch', 'lighter', 56.76932576362212], ['totem pole', 'pole', 46.612185110561484], ['tow truck', 'crane', 41.14048554381985], ['toyshop', 'tobacco shop', 32.771335786304924], ['tractor', 'harvester', 29.482276894414145], ['trailer truck', 'moving van', 24.706998826559754], ['tray', 'backpack', 15.724673783598371], ['trench coat', 'suit', 47.25856093906426], ['tricycle', 'folding chair', 51.39589561373067], ['trimaran', 'catamaran', 22.3432651280432], ['tripod', 'schooner', 54.92037026743811], ['triumphal arch', 'monastery', 40.06770697928851], ['trolleybus', 'minibus', 30.701204615515497], ['trombone', 'cornet', 32.82054216921546], ['tub', 'bathtub', 16.951423805774215], ['turnstile', 'sliding door', 63.8902753567102], ['typewriter keyboard', 'space bar', 21.416328318653125], ['umbrella', 'parachute', 40.21868929676849], ['unicycle', 'mountain bike', 40.76010586920439], ['upright', 'grand piano', 30.031540630097343], ['vacuum', 'lawn mower', 68.64477743244565], ['vase', 'whiskey jug', 60.48773926642076], ['vault', 'altar', 34.53101314964021], ['velvet', 'handkerchief', 42.06511472649427], ['vending machine', 'tobacco shop', 33.653319278155905], ['vestment', 'altar', 38.92552703912225], ['viaduct', 'lakeside', 48.363835113388184], ['violin', 'cello', 31.501572342640195], ['volleyball', 'basketball', 41.629351780774826], ['waffle iron', 'carbonara', 55.415524639704984], ['wall clock', 'analog clock', 16.509403208084457], ['wallet', 'binder', 37.541620783013954], ['wardrobe', 'sliding door', 27.571849941193417], ['warplane', 'airliner', 30.145205231305027], ['washbasin', 'bathtub', 38.96337579471666], ['washer', 'Crock Pot', 51.059638364581815], ['water bottle', 'pop bottle', 27.81417875902617], ['water jug', 'pitcher', 29.683772381108124], ['water tower', 'mosque', 59.34531947305849], ['whiskey jug', 'pot', 43.573065769427735], ['whistle', 'carton', 15.297686951765575], ['wig', 'groom', 65.59931789136607], ['window screen', 'sliding door', 30.00443692775391], ['window shade', 'shoji', 29.342636136007894], ['Windsor tie', 'suit', 31.61707456892359], ['wine bottle', 'red wine', 17.07245678036931], ['wing', 'airliner', 50.62216668383155], ['wok', 'frying pan', 31.63438885304201], ['wooden spoon', 'ladle', 50.45412907409066], ['wool', 'dishrag', 46.56170825669146], ['worm fence', 'stone wall', 33.22389423950028], ['wreck', 'harvester', 55.07415100911679], ['yawl', 'catamaran', 27.542372363153923], ['yurt', 'dome', 51.4125803473114], ['web site', 'screen', 40.981302603576886], ['comic book', 'book jacket', 26.3766510794674], ['crossword puzzle', 'menu', 50.82210135687155], ['street sign', 'tobacco shop', 41.958677763347566], ['traffic light', 'street sign', 46.78904619406128], ['book jacket', 'comic book', 26.3766510794674], ['menu', 'brass', 41.71193910929834], ['plate', 'carbonara', 43.90882443520038], ['guacamole', 'gyromitra', 44.291026950832546], ['consomme', 'soup bowl', 27.428323449421654], ['hot pot', 'consomme', 30.729245012258882], ['trifle', 'stone wall', 59.376759842340675], ['ice cream', 'dough', 36.998980533886], ['ice lolly', 'pill bottle', 76.29637421758652], ['French loaf', 'hotdog', 54.0078028940032], ['bagel', 'earthstar', 46.550674882661475], ['pretzel', 'sidewinder', 42.24460833755889], ['cheeseburger', 'meat loaf', 43.929122712223894], ['hotdog', 'pizza', 47.144874166299886], ['mashed potato', 'cauliflower', 33.19454300855602], ['head cabbage', 'wool', 52.90255358469239], ['broccoli', 'coral fungus', 57.604541514753855], ['cauliflower', 'mashed potato', 33.19454300855602], ['zucchini', 'cucumber', 67.02484672212985], ['spaghetti squash', 'carbonara', 38.407206685602965], ['acorn squash', 'butternut squash', 55.97634824191375], ['butternut squash', 'gyromitra', 45.59681813700107], ['cucumber', 'feather boa', 61.10540769530447], ['artichoke', 'cardoon', 47.17056556645654], ['bell pepper', "yellow lady's slipper", 56.56725851875911], ['cardoon', 'sea urchin', 42.66654380144867], ['mushroom', 'agaric', 38.810527954127664], ['Granny Smith', 'earthstar', 52.988676193008445], ['strawberry', 'pomegranate', 42.848094806136075], ['orange', 'lemon', 31.253908935211108], ['lemon', 'orange', 31.253908935211108], ['fig', 'pomegranate', 51.69458697257988], ['pineapple', 'cardoon', 44.035294682043116], ['banana', 'butternut squash', 64.84698034122329], ['jackfruit', 'strawberry', 51.2258255675604], ['custard apple', 'earthstar', 58.334865586274816], ['pomegranate', 'strawberry', 42.848094806136075], ['hay', 'megalith', 53.245868934535665], ['carbonara', 'spaghetti squash', 38.407206685602965], ['chocolate sauce', 'ice cream', 42.18490154034676], ['dough', 'ice cream', 36.998980533886], ['meat loaf', 'cheeseburger', 43.929122712223894], ['pizza', 'potpie', 35.38706004395798], ['potpie', 'pizza', 35.38706004395798], ['burrito', 'bluetick', 58.17940372312289], ['red wine', 'wine bottle', 17.07245678036931], ['espresso', 'chocolate sauce', 42.87866835846599], ['cup', 'coffee mug', 34.77271521800983], ['eggnog', 'espresso', 46.282167867333094], ['alp', 'valley', 24.653393172581232], ['bubble', 'Petri dish', 47.17110901166757], ['cliff', 'promontory', 33.63025632707845], ['coral reef', 'sea slug', 32.6858941407643], ['geyser', 'window screen', 46.98921501924671], ['lakeside', 'dock', 29.90675919596553], ['promontory', 'seashore', 29.481354591583603], ['sandbar', 'seashore', 25.69905695391005], ['seashore', 'sandbar', 25.69905695391005], ['valley', 'alp', 24.653393172581232], ['volcano', 'valley', 55.58231332616069], ['ballplayer', 'puck', 49.903706102978965], ['groom', 'gown', 27.881970821213482], ['scuba diver', 'snorkel', 47.666791866822756], ['rapeseed', 'monarch', 39.90250110865937], ['daisy', 'sea anemone', 38.35793976439233], ["yellow lady's slipper", 'hip', 43.27372875759538], ['corn', 'ear', 19.72982568925827], ['acorn', 'ladybug', 49.60812252355959], ['hip', 'ladybug', 42.50306026436141], ['buckeye', 'sea cucumber', 54.185418428877156], ['coral fungus', 'gyromitra', 40.012485844264916], ['agaric', 'bolete', 38.68633725787035], ['gyromitra', 'coral fungus', 40.012485844264916], ['stinkhorn', 'strawberry', 55.914655775286235], ['earthstar', 'bagel', 46.550674882661475], ['hen-of-the-woods', 'coral fungus', 41.22301585440847], ['bolete', 'agaric', 38.68633725787035], ['ear', 'corn', 19.72982568925827], ['toilet tissue', 'paper towel', 24.520790465216706]]
[0.15383747 0.04181266]

# latent_space.py

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import seaborn as sns
import sklearn.decomposition as decomp
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from google.colab import files
# from google.colab import drive

# drive.mount('/content/gdrive')

# data_dir = pathlib.Path('/content/gdrive/My Drive/Inception_generation',  fname='Combined')
# image_count = len(list(data_dir.glob('*.png')))

files.upload() # upload files
!unzip dalmatians.zip

# dataset directory specification
data_dir = pathlib.Path('dalmatians',  fname='Combined')

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}")

class ImageDataset(Dataset):
    """
    Creates a dataset from images classified by folder name.  Random
    sampling of images to prevent overfitting
    """

    def __init__(self, img_dir, image_type='.png'):
        self.image_name_ls = list(img_dir.glob('*/*' + image_type))
        self.img_labels = [item.name for item in data_dir.glob('*/*')]
        self.img_dir = img_dir

    def __len__(self):
        return len(self.image_name_ls)

    def __getitem__(self, index):
        # path to image
        img_path = os.path.join(self.image_name_ls[index])
        image = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB) # convert image to tensor of ints 
        image = image / 255. # convert ints to floats in range [0, 1]
        image = torchvision.transforms.Resize(size=[299, 299])(image)

        # assign label
        label = os.path.basename(img_path)
        return image, label


googlenet = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False, init_weights=True).to(device)
images = ImageDataset(data_dir, image_type='.jpg')

resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).to(device)
googlenet.eval()
resnet.eval()

# network = NewGoogleNet(googlenet)
# network.eval()
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

class NewResNet(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        # x = self.model.avgpool(x)
        # x = self.model.flatten(x, 1)
        # x = self.model.fc(x)
        return x

class NewGoogleNet(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
            # N x 3 x 224 x 224
            x = self.model.conv1(x)
            # N x 64 x 112 x 112
            x = self.model.maxpool1(x)
            # N x 64 x 56 x 56
            x = self.model.conv2(x)
            # # N x 64 x 56 x 56
            x = self.model.conv3(x)
            # # N x 192 x 56 x 56
            x = self.model.maxpool2(x)
            # # N x 192 x 28 x 28
            x = self.model.inception3a(x)
            # # N x 256 x 28 x 28
            # x = self.model.inception3b(x)
            # # N x 480 x 28 x 28
            # x = self.model.maxpool3(x)
            # # N x 480 x 14 x 14
            # x = self.model.inception4a(x)
            # N x 512 x 14 x 14
            # x = self.model.inception4b(x)
            # N x 512 x 14 x 14
            # x = self.model.inception4c(x)
            # # N x 512 x 14 x 14
            # x = self.model.inception4d(x)
            # # N x 528 x 14 x 14
            # x = self.model.inception4e(x)
            # # N x 832 x 14 x 14
            # x = self.model.maxpool4(x)
            # # N x 832 x 7 x 7
            # x = self.model.inception5a(x)
            # # N x 832 x 7 x 7
            # x = self.model.inception5b(x)
            # # N x 1024 x 7 x 7
            # x = self.model.avgpool(x)
            # # N x 1024 x 1 x 1
            # x = torch.flatten(x, 1)
            # # N x 1024
            # x = self.model.dropout(x)
            # x = self.model.fc(x)
            # N x 1000 (num_classes)
            return x

resnet2 = NewResNet(resnet)
resnet2.eval()

def train(model, input_tensor, target_output):
    """
    Train a single minibatch

    Args:
        input_tensor: torch.Tensor object 
        output_tensor: torch.Tensor object
        optimizer: torch.optim object
        minibatch_size: int, number of examples per minibatch
        model: torch.nn

    Returns:
        output: torch.Tensor of model predictions
        loss.item(): float of loss for that minibatch

    """
    # self.model.train()
    output = model(input_tensor)
    loss = torch.sum(torch.abs(output - target_output)**2)
    optimizer.zero_grad() # prevents gradients from adding between minibatches
    loss.backward()
    optimizer.step()
    return 


optimizer = torch.optim.SGD(resnet.parameters(), lr=0.00001)
for i, image in enumerate(images):
    if i == 0:
        print (i)
        image = image[0].reshape(1, 3, 299, 299).to(device)
        # target_output = googlenet(image).detach().to(device)
        target_tensor = resnet2(image)
        # image2 = torchvision.transforms.Resize(size=[470, 470])(image)
        # target_tensor2 = resnet2(image2).detach().to(device)
        # image3 = torchvision.transforms.Resize(size=[570, 570])(image)
        # target_tensor3 = resnet2(image3).detach().to(device)
        break

print (torch.argmax(target_tensor))
target_tensor = target_tensor.detach().to(device)
# resnet.eval()

plt.figure(figsize=(10, 10))
image_width = len(image[0][0])
target_input = image.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
plt.axis('off')
plt.imshow(target_input)
plt.show()
plt.close()


def random_crop(input_image, size):
    """
    Crop an image with a starting x, y coord from a uniform distribution

    Args:
        input_image: torch.tensor object to be cropped
        size: int, size of the desired image (size = length = width)

    Returns:
        input_image_cropped: torch.tensor
        crop_height: starting y coordinate
        crop_width: starting x coordinate
    """

    image_width = len(input_image[0][0])
    image_height = len(input_image[0])
    crop_width = random.randint(0, image_width - size)
    crop_height = random.randint(0, image_width - size)
    input_image_cropped = input_image[:, :, crop_height:crop_height + size, crop_width: crop_width + size]

    return input_image_cropped, crop_height, crop_width


def octave(single_input, target_output, iterations, learning_rates, sigmas, size, pad=False, crop=True):
    """
    Perform an octave (scaled) gradient descent on the input.

    Args;
        single_input: torch.tensor of the input
        target_output: torch.tensor of the desired output category
        iterations: int, the number of iterations desired
        learning_rates: arr[int, int], pair of integers corresponding to start and end learning rates
        sigmas: arr[int, int], pair of integers corresponding to the start and end Gaussian blur sigmas
        size: int, desired dimension of output image (size = length = width)

    kwargs:
        pad: bool, if True then padding is applied at each iteration of the octave
        crop: bool, if True then gradient descent is applied to cropped sections of the input

    Returns:
        single_input: torch.tensor of the transformed input
    """

    start_lr, end_lr = learning_rates
    start_sigma, end_sigma = sigmas

    for i in range(iterations):
        if crop:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), size)
        else:
            cropped_input, crop_height, crop_width = random_crop(single_input.detach(), len(single_input[0][0]))
            size = len(single_input[0][0])
        single_input = single_input.detach() # remove the gradient for the input (if present)
        input_grad = layer_gradient(resnet, cropped_input, target_output) # compute input gradient
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] -= (start_lr*(iterations-i)/iterations + end_lr*i/iterations)* input_grad # gradient descent step
        single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size] = torchvision.transforms.functional.gaussian_blur(single_input[:, :, crop_height:crop_height+size, crop_width:crop_width+size], 3, sigma=(start_sigma*(iterations-i)/iterations + end_sigma*i/iterations))

    return single_input


def generate_singleinput(model, input_tensors, output_tensors, index, count, random_input=True):
    """
    Generates an input for a given output

    Args:
        input_tensor: torch.Tensor object, minibatch of inputs
        output_tensor: torch.Tensor object, minibatch of outputs
        index: int, target class index to generate
        cout: int, time step

    kwargs: 
        random_input: bool, if True then a scaled random normal distributionis used

    returns:
        None (saves .png image)
    """

    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    class_index = index

    if random_input:
        single_input = (torch.randn(1, 3, 299, 299))/20 + 0.7 # scaled normal distribution initialization

    else:
        single_input = input_tensors[0]
 
    single_input = single_input.to(device)
    original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
    single_input = single_input.reshape(1, 3, 299, 299)
    original_input = torch.clone(single_input).reshape(3, 299, 299).permute(1, 2, 0).cpu().detach().numpy()
    target_output = torch.tensor([class_index], dtype=int)
 
    single_input = octave(single_input, target_output, 220, [2.5, 1.5], [2.4, 0.4], 0, pad=False, crop=False)
    # single_input = torchvision.transforms.Resize([490, 490])(single_input)
    # single_input = octave(single_input, target_output, 500, [1.5, 0.3], [1.5, 0.4], 470, pad=False, crop=True) 
    # single_input = torchvision.transforms.Resize([600, 600])(single_input)
    # single_input = octave(single_input, target_output, 100, [1.5, 0.4], [1.5, 0.4], 570, pad=False, crop=True)

    output = resnet(single_input)
    plt.figure(figsize=(10, 10))
    image_width = len(single_input[0][0])
    target_input = single_input.reshape(3, image_width, image_width).permute(1, 2, 0).cpu().detach().numpy()
    plt.axis('off')
    plt.imshow(target_input)
    plt.show()
    plt.close()
    return single_input


def layer_gradient(model, input_tensor, desired_output):
    """
    Compute the gradient of the output (logits) with respect to the input 
    using an L1 metric to maximize the target classification.

    Args:
        model: torch.nn.model
        input_tensor: torch.tensor object corresponding to the input image
        true_output: torch.tensor object of the desired classification label

    Returns:
        gradient: torch.tensor.grad on the input tensor after backpropegation

    """
    input_tensor.requires_grad = True
    output = resnet2(input_tensor)
    loss = 0.0002*torch.sum(torch.abs(target_tensor - output))
    # elif len(input_tensor[0][0]) == 470:
    #     loss = 0.1*torch.sum(torch.abs(target_tensor2 - output))
    # else:
    #     loss = 0.1*torch.sum(torch.abs(target_tensor3 - output))

    loss.backward()
    gradient = input_tensor.grad

    return gradient

generate_singleinput(resnet2, [], [], 0, 0)


# x, y, labels_arr = [], [], []
# for i, image in enumerate(images):
#     print (i)
#     label = image[1]
#     image = image[0].reshape(1, 3, 299, 299).to(device)
#     output = network(image)
#     x.append(float(torch.mean(output[0, 0, :, :])))
#     y.append(float(torch.mean(output[0, 5, :, :])))
#     i = 11
#     while label[i] not in ',.':
#         i += 1
#     labels_arr.append(label[11:i])
    
# plt.figure(figsize=(18, 18))
# plt.scatter(x, y)
# for i, label in enumerate(labels_arr):
#     plt.annotate(label, (x[i], y[i]))
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 4')
# plt.show()
# plt.close()

# sns.jointplot(x, y)
# plt.show()
# plt.close()


# outputs, labels_arr = [], []
# count = 0
# for i, image in enumerate(images):
#     label = image[1]
#     image = image[0].reshape(1, 3, 299, 299).to(device)
#     output = googlenet(image)
#     output = output.detach().cpu().numpy()
#     outputs.append(output)
#     j = 11
#     while label[j] not in ',.':
#         j += 1
#     labels_arr.append(label[12:j])
#     if i == np.argmax(output):
#         count += 1
#     print (i, count)

# print (count)

# for i, output in enumerate(outputs):
#     print (labels_arr[i], np.argmin(output[0]))
#     print (min(output[0]))

# basis_distances = []
# for i, output in enumerate(outputs):
#     if np.argsort(output[0])[-1] == i:
#         index = np.argsort(output[0])[-2]
#     else:
#         index = np.argsort(output[0])[-1]
#     print (i, index)
#     print (output[0][index])
#     basis_distances.append([labels_arr[i], labels_arr[index], output[0][index]])
# print (basis_distances)

# l2_distances = []
# for i, output in enumerate(outputs):
#     distances = [(np.sum((output - i)**2))**0.5 for i in outputs]
#     index = np.argsort(distances)[1] # find the second-smallest distance (includes the same point)
#     l2_distances.append([labels_arr[i], labels_arr[index], distances[index]])
# print (l2_distances)

# outputs = torch.tensor(outputs)
# outputs = outputs.reshape(len(outputs), 1000)
# pca = decomp.PCA(n_components=2)
# pca.fit(outputs)
# print (pca.explained_variance_ratio_)
# arr = pca.transform(outputs)
# x, y = [i[0] for i in arr], [i[1] for i in arr]
# plt.figure(figsize=(18, 18))
# plt.scatter(x, y)
# for i, label in enumerate(labels_arr):
#     plt.annotate(label, (x[i], y[i]))

# plt.xlabel('Feature 0')
# plt.ylabel('Feature 4')
# plt.title('GoogleNet Layer 5a Embedding')
# plt.show()
# plt.close()

# sns.jointplot(x, y)
# plt.show()
# plt.close()

