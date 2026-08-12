export const imageDatasets = [
  /* Cervical Cancer */
  { category: "Cervical Cancer", name: "SIPaKMeD", classes: 5, size: "4,049 cells", shape: [224, 224, 3], url: "https://www.cs.uoi.gr/~marina/sipakmed.html", compatible: "cnn/transformer/unet" },
  { category: "Cervical Cancer", name: "LBC Cervical Cancer", classes: 4, size: "Pap smear LBC", shape: [224, 224, 3], url: "https://data.mendeley.com/datasets/zddtpgzv63/4", compatible: "cnn/transformer/unet" },
  { category: "Cervical Cancer", name: "APCData Cytology", classes: 6, size: "425 images", shape: [224, 224, 3], url: "https://data.mendeley.com/datasets/ytd568rh3p/1", compatible: "cnn/transformer/unet" },

  /* Skin Cancer */
  { category: "Skin Cancer", name: "ISIC 2019", classes: 8, size: "25,331 images", shape: [224, 224, 3], url: "https://challenge.isic-archive.com/data/#2019", compatible: "cnn/transformer/unet" },
  { category: "Skin Cancer", name: "HAM10000", classes: 7, size: "10,015 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000", compatible: "cnn/transformer/unet" },
  { category: "Skin Cancer", name: "DermaMNIST", classes: 7, size: "10,015 images", shape: [28, 28, 3], url: "https://medmnist.com/", compatible: "cnn/transformer" },

  /* Blood Classifications */
  { category: "Blood Classifications", name: "Blood Cell (BCCD)", classes: 4, size: "12,444 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/paultimothymooney/blood-cells", compatible: "cnn/transformer/unet" },
  { category: "Blood Classifications", name: "BloodMNIST", classes: 8, size: "17,092 images", shape: [28, 28, 3], url: "https://medmnist.com/", compatible: "cnn/transformer" },

  /* Brain Tumor */
  { category: "Brain Tumor", name: "BraTS 2021", classes: 4, size: "8,000+ MRI scans", shape: [240, 240, 4], url: "https://www.med.upenn.edu/cbica/brats2021/", compatible: "cnn/transformer/unet" },
  { category: "Brain Tumor", name: "Brain Tumor MRI", classes: 4, size: "7,023 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset", compatible: "cnn/transformer/unet" },

  /* Breast Cancer */
  { category: "Breast Cancer", name: "BreakHis", classes: 8, size: "7,909 images", shape: [224, 224, 3], url: "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/", compatible: "cnn/transformer/unet" },
  { category: "Breast Cancer", name: "CBIS-DDSM", classes: 2, size: "10,239 images", shape: [224, 224, 1], url: "https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM", compatible: "cnn/transformer" },

  /* Natural/Other */
  { category: "Natural Benchmarks", name: "CIFAR-10", classes: 10, size: "60k 32×32", shape: [32, 32, 3], url: "https://www.cs.toronto.edu/~kriz/cifar.html", compatible: "cnn/transformer" },
  { category: "Natural Benchmarks", name: "ImageNet Mini", classes: 1000, size: "subset", shape: [224, 224, 3], url: "https://www.image-net.org/", compatible: "cnn/transformer" },
];

export const csvDatasets = [
  { category: "Classical ML", name: "Breast Cancer Wisconsin", target: "diagnosis", features: 30, samples: 569, url: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic", compatible: "ml" },
  { category: "Classical ML", name: "Heart Disease", target: "target", features: 13, samples: 303, url: "https://archive.ics.uci.edu/dataset/45/heart+disease", compatible: "ml" },
  { category: "Classical ML", name: "Pima Diabetes", target: "Outcome", features: 8, samples: 768, url: "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database", compatible: "ml" },
  { category: "Classical ML", name: "Titanic", target: "Survived", features: 11, samples: 891, url: "https://www.kaggle.com/c/titanic", compatible: "ml" },
  { category: "Classical ML", name: "Iris", target: "species", features: 4, samples: 150, url: "https://archive.ics.uci.edu/dataset/53/iris", compatible: "ml" },
];

export const augmentations = [
  "Resize", "RandomResizedCrop", "HorizontalFlip", "VerticalFlip",
  "Rotation", "ColorJitter", "GaussianBlur", "RandomErasing",
  "AutoAugment", "RandAugment", "MixUp", "CutMix", "Normalize",
];
