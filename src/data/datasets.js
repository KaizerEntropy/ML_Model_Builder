export const imageDatasets = [
  /* Classification - Cervical Cancer */
  { category: "Cervical Cancer", name: "SIPaKMeD", classes: 5, size: "4,049 cells", shape: [224, 224, 3], url: "https://www.cs.uoi.gr/~marina/sipakmed.html", compatible: "cnn/transformer" },
  { category: "Cervical Cancer", name: "LBC Cervical Cancer", classes: 4, size: "Pap smear LBC", shape: [224, 224, 3], url: "https://data.mendeley.com/datasets/zddtpgzv63/4", compatible: "cnn/transformer" },
  { category: "Cervical Cancer", name: "APCData Cytology", classes: 6, size: "425 images", shape: [224, 224, 3], url: "https://data.mendeley.com/datasets/ytd568rh3p/1", compatible: "cnn/transformer" },

  /* Classification - Skin Cancer */
  { category: "Skin Cancer", name: "ISIC 2019", classes: 8, size: "25,331 images", shape: [224, 224, 3], url: "https://challenge.isic-archive.com/data/#2019", compatible: "cnn/transformer" },
  { category: "Skin Cancer", name: "ISIC 2018", classes: 7, size: "10,015 images", shape: [224, 224, 3], url: "https://challenge.isic-archive.com/data/#2018", compatible: "cnn/transformer" },
  { category: "Skin Cancer", name: "ISIC 2017", classes: 3, size: "2,000 images", shape: [224, 224, 3], url: "https://challenge.isic-archive.com/data/#2017", compatible: "cnn/transformer" },
  { category: "Skin Cancer", name: "HAM10000", classes: 7, size: "10,015 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000", compatible: "cnn/transformer" },
  { category: "Skin Cancer", name: "DermaMNIST", classes: 7, size: "10,015 images", shape: [28, 28, 3], url: "https://medmnist.com/", compatible: "cnn/transformer" },
  { category: "Skin Cancer", name: "Milk10k", classes: 2, size: "10,000 images", shape: [224, 224, 3], url: "#", compatible: "cnn/transformer" },

  /* Classification - Blood Classifications */
  { category: "Blood Classifications", name: "Blood Cell (BCCD)", classes: 4, size: "12,444 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/paultimothymooney/blood-cells", compatible: "cnn/transformer" },
  { category: "Blood Classifications", name: "RaabinWBC", classes: 5, size: "1,145 images", shape: [224, 224, 3], url: "https://raabin.iau.ir/wbc/", compatible: "cnn/transformer" },
  { category: "Blood Classifications", name: "BloodMNIST", classes: 8, size: "17,092 images", shape: [28, 28, 3], url: "https://medmnist.com/", compatible: "cnn/transformer" },

  /* Classification - Brain Tumor */
  { category: "Brain Tumor", name: "Brain Tumor MRI", classes: 4, size: "7,023 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset", compatible: "cnn/transformer" },

  /* Classification - Breast Cancer */
  { category: "Breast Cancer", name: "BreakHis", classes: 8, size: "7,909 images", shape: [224, 224, 3], url: "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/", compatible: "cnn/transformer" },
  { category: "Breast Cancer", name: "BACH", classes: 4, size: "400 images", shape: [224, 224, 3], url: "https://iciar2018-challenge.grand-challenge.org/", compatible: "cnn/transformer" },
  { category: "Breast Cancer", name: "CBIS-DDSM", classes: 2, size: "10,239 images", shape: [224, 224, 1], url: "https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM", compatible: "cnn/transformer" },
  { category: "Breast Cancer", name: "BreastMNIST", classes: 2, size: "780 images", shape: [28, 28, 1], url: "https://medmnist.com/", compatible: "cnn/transformer" },

  /* Classification - Chest XRay */
  { category: "Chest XRay", name: "NIH Chest X-ray", classes: 14, size: "112,120 images", shape: [224, 224, 1], url: "https://nihcc.app.box.com/v/ChestXray-NIHCC", compatible: "cnn/transformer" },
  { category: "Chest XRay", name: "MosMed", classes: 2, size: "1,110 images", shape: [224, 224, 1], url: "https://mosmed.ai/datasets/covid19_1110", compatible: "cnn/transformer" },
  { category: "Chest XRay", name: "QaTa-COV19", classes: 2, size: "6,200 images", shape: [224, 224, 1], url: "https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset", compatible: "cnn/transformer" },
  { category: "Chest XRay", name: "ChestMNIST", classes: 14, size: "112,120 images", shape: [28, 28, 1], url: "https://medmnist.com/", compatible: "cnn/transformer" },
  { category: "Chest XRay", name: "PneumoniaMNIST", classes: 2, size: "5,856 images", shape: [28, 28, 1], url: "https://medmnist.com/", compatible: "cnn/transformer" },

  /* Classification - Other Medical */
  { category: "Other Medical", name: "PathMNIST", classes: 9, size: "107,180 images", shape: [28, 28, 3], url: "https://medmnist.com/", compatible: "cnn/transformer" },
  { category: "Other Medical", name: "OCTMNIST", classes: 4, size: "109,309 images", shape: [28, 28, 1], url: "https://medmnist.com/", compatible: "cnn/transformer" },
  { category: "Other Medical", name: "RetinaMNIST", classes: 5, size: "1,600 images", shape: [28, 28, 3], url: "https://medmnist.com/", compatible: "cnn/transformer" },
  { category: "Other Medical", name: "TissueMNIST", classes: 8, size: "236,386 images", shape: [28, 28, 1], url: "https://medmnist.com/", compatible: "cnn/transformer" },
  { category: "Other Medical", name: "OrganAMNIST", classes: 11, size: "58,850 images", shape: [28, 28, 1], url: "https://medmnist.com/", compatible: "cnn/transformer" },

  /* Classification - Natural Benchmarks */
  { category: "Natural Benchmarks", name: "CIFAR-10", classes: 10, size: "60k 32×32", shape: [32, 32, 3], url: "https://www.cs.toronto.edu/~kriz/cifar.html", compatible: "cnn/transformer" },
  { category: "Natural Benchmarks", name: "CIFAR-100", classes: 100, size: "60k 32×32", shape: [32, 32, 3], url: "https://www.cs.toronto.edu/~kriz/cifar.html", compatible: "cnn/transformer" },
  { category: "Natural Benchmarks", name: "Food101", classes: 101, size: "101k images", shape: [224, 224, 3], url: "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/", compatible: "cnn/transformer" },
  { category: "Natural Benchmarks", name: "Caltech-101", classes: 101, size: "9,146 images", shape: [224, 224, 3], url: "https://data.caltech.edu/records/mzrjq-6wc69", compatible: "cnn/transformer" },
  { category: "Natural Benchmarks", name: "FashionMNIST", classes: 10, size: "70k 28×28", shape: [28, 28, 1], url: "https://github.com/zalandoresearch/fashion-mnist", compatible: "cnn/transformer" },
  { category: "Natural Benchmarks", name: "ImageNet Mini", classes: 1000, size: "subset", shape: [224, 224, 3], url: "https://www.image-net.org/", compatible: "cnn/transformer" },

  /* Segmentation - Medical Segmentation */
  { category: "Medical Segmentation", name: "BraTS 2021", classes: 4, size: "8,000+ MRI scans", shape: [240, 240, 4], url: "https://www.med.upenn.edu/cbica/brats2021/", compatible: "unet" },
  { category: "Medical Segmentation", name: "Kvasir-SEG", classes: 2, size: "1,000 images", shape: [256, 256, 3], url: "https://datasets.simula.no/kvasir-seg/", compatible: "unet" },
  { category: "Medical Segmentation", name: "MoNuSAC", classes: 5, size: "300 images", shape: [256, 256, 3], url: "https://monusac-2020.grand-challenge.org/", compatible: "unet" },
  { category: "Medical Segmentation", name: "DRIVE Retina", classes: 2, size: "40 images", shape: [584, 565, 3], url: "https://drive.grand-challenge.org/", compatible: "unet" },

  /* Segmentation - Natural Segmentation */
  { category: "Natural Segmentation", name: "Cityscapes", classes: 30, size: "5,000 images", shape: [256, 256, 3], url: "https://www.cityscapes-dataset.com/", compatible: "unet" },
  { category: "Natural Segmentation", name: "CamVid", classes: 32, size: "700 images", shape: [720, 960, 3], url: "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/", compatible: "unet" },
  { category: "Natural Segmentation", name: "PASCAL VOC 2012", classes: 21, size: "2,913 images", shape: [256, 256, 3], url: "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/", compatible: "unet" },

  /* Object Detection */
  { category: "Object Detection", name: "COCO 2017", classes: 80, size: "118k images", shape: [640, 640, 3], url: "https://cocodataset.org/", compatible: "od" },
  { category: "Object Detection", name: "PASCAL VOC 2007+2012", classes: 20, size: "16k images", shape: [416, 416, 3], url: "http://host.robots.ox.ac.uk/pascal/VOC/", compatible: "od" },
  { category: "Object Detection", name: "Open Images V7", classes: 600, size: "1.7M images", shape: [640, 640, 3], url: "https://storage.googleapis.com/openimages/web/index.html", compatible: "od" },
  { category: "Object Detection", name: "WiderPerson", classes: 5, size: "13,382 images", shape: [416, 416, 3], url: "http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/", compatible: "od" },
  { category: "Object Detection", name: "Global Wheat Detection", classes: 1, size: "3,422 images", shape: [1024, 1024, 3], url: "https://www.kaggle.com/c/global-wheat-detection", compatible: "od" }
];

export const imageFilters = ["None", "Sobel", "Gaussian", "Laplace", "Canny", "Median"];


export const csvDatasets = [
  /* Classification */
  { category: "Classification", name: "Breast Cancer Wisconsin", target: "diagnosis", features: 30, samples: 569, url: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic", compatible: "ml" },
  { category: "Classification", name: "Heart Disease", target: "target", features: 13, samples: 303, url: "https://archive.ics.uci.edu/dataset/45/heart+disease", compatible: "ml" },
  { category: "Classification", name: "Pima Diabetes", target: "Outcome", features: 8, samples: 768, url: "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database", compatible: "ml" },
  { category: "Classification", name: "Titanic", target: "Survived", features: 11, samples: 891, url: "https://www.kaggle.com/c/titanic", compatible: "ml" },
  { category: "Classification", name: "Iris", target: "species", features: 4, samples: 150, url: "https://archive.ics.uci.edu/dataset/53/iris", compatible: "ml" },
  { category: "Classification", name: "Wine Quality", target: "quality", features: 11, samples: 4898, url: "https://archive.ics.uci.edu/dataset/186/wine+quality", compatible: "ml" },
  { category: "Classification", name: "Banknote Authentication", target: "class", features: 4, samples: 1372, url: "https://archive.ics.uci.edu/dataset/267/banknote+authentication", compatible: "ml" },
  { category: "Classification", name: "Dry Bean", target: "Class", features: 16, samples: 13611, url: "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset", compatible: "ml" },

  /* Regression */
  { category: "Regression", name: "Boston Housing", target: "MEDV", features: 13, samples: 506, url: "https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html", compatible: "ml" },
  { category: "Regression", name: "California Housing", target: "MedHouseVal", features: 8, samples: 20640, url: "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html", compatible: "ml" },
  { category: "Regression", name: "Auto MPG", target: "mpg", features: 7, samples: 398, url: "https://archive.ics.uci.edu/dataset/9/auto+mpg", compatible: "ml" },
  { category: "Regression", name: "Bike Sharing", target: "cnt", features: 16, samples: 17379, url: "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset", compatible: "ml" },
  { category: "Regression", name: "Medical Cost", target: "charges", features: 6, samples: 1338, url: "https://www.kaggle.com/datasets/mirichoi0218/insurance", compatible: "ml" },
];

export const augmentations = [
  "Resize", "RandomResizedCrop", "HorizontalFlip", "VerticalFlip",
  "Rotation", "ColorJitter", "GaussianBlur", "RandomErasing",
  "AutoAugment", "RandAugment", "MixUp", "CutMix", "Normalize",
];
