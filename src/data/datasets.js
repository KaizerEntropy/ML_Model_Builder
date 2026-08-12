/* ── Datasets: image (CNN) and CSV (classical ML) ── */

export const imageDatasets = [
  /* Medical: cervical */
  { category: "Medical: Cervical", name: "SIPaKMeD", classes: 5, size: "4,049 cells", shape: [224, 224, 3], url: "https://www.cs.uoi.gr/~marina/sipakmed.html" },
  { category: "Medical: Cervical", name: "Mendeley LBC Cervical Cancer", classes: 4, size: "Pap smear LBC", shape: [224, 224, 3], url: "https://data.mendeley.com/datasets/zddtpgzv63/4" },
  { category: "Medical: Cervical", name: "APCData Cervical Cytology", classes: 6, size: "425 images", shape: [224, 224, 3], url: "https://data.mendeley.com/datasets/ytd568rh3p/1" },

  /* Medical: breast */
  { category: "Medical: Breast", name: "BACH", classes: 4, size: "histology", shape: [224, 224, 3], url: "https://zenodo.org/records/3632035" },
  { category: "Medical: Breast", name: "BreakHis", classes: 8, size: "9,109 images", shape: [224, 224, 3], url: "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/" },

  /* Medical: dermatology */
  { category: "Medical: Dermatology", name: "ISIC 2019 Skin Lesion", classes: 8, size: "25,331 images", shape: [224, 224, 3], url: "https://challenge.isic-archive.com/data/#2019" },
  { category: "Medical: Dermatology", name: "HAM10000", classes: 7, size: "10,015 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000" },
  { category: "Medical: Dermatology", name: "DermaMNIST (MedMNIST)", classes: 7, size: "10,015 images", shape: [28, 28, 3], url: "https://medmnist.com/" },

  /* Medical: ophthalmology */
  { category: "Medical: Ophthalmology", name: "APTOS 2019 Diabetic Retinopathy", classes: 5, size: "5,590 images", shape: [224, 224, 3], url: "https://www.kaggle.com/c/aptos2019-blindness-detection" },
  { category: "Medical: Ophthalmology", name: "MESSIDOR-2", classes: 5, size: "1,748 images", shape: [224, 224, 3], url: "https://www.adcis.net/en/third-party/messidor2/" },

  /* Medical: chest */
  { category: "Medical: Chest", name: "ChestX-ray14 (NIH)", classes: 14, size: "112,120 images", shape: [224, 224, 3], url: "https://nihcc.app.box.com/v/ChestXray-NIHCC" },
  { category: "Medical: Chest", name: "COVID-19 Radiography", classes: 4, size: "21,165 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database" },

  /* Medical: brain */
  { category: "Medical: Brain", name: "Brain Tumor MRI", classes: 4, size: "7,023 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset" },

  /* Medical: histopathology */
  { category: "Medical: Histopathology", name: "LC25000 Lung & Colon", classes: 5, size: "25,000 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images" },
  { category: "Medical: Histopathology", name: "PathMNIST (MedMNIST)", classes: 9, size: "107,180 images", shape: [28, 28, 3], url: "https://medmnist.com/" },

  /* Medical: hematology */
  { category: "Medical: Hematology", name: "Blood Cell Images (BCCD)", classes: 4, size: "12,444 images", shape: [224, 224, 3], url: "https://www.kaggle.com/datasets/paultimothymooney/blood-cells" },

  /* Medical: gastrointestinal */
  { category: "Medical: Gastrointestinal", name: "Kvasir GI Endoscopy", classes: 8, size: "8,000 images", shape: [224, 224, 3], url: "https://datasets.simula.no/kvasir/" },

  /* Natural image benchmarks */
  { category: "Natural", name: "CIFAR-10", classes: 10, size: "60k 32×32", shape: [32, 32, 3], url: "https://www.cs.toronto.edu/~kriz/cifar.html" },
  { category: "Natural", name: "CIFAR-100", classes: 100, size: "60k 32×32", shape: [32, 32, 3], url: "https://www.cs.toronto.edu/~kriz/cifar.html" },
  { category: "Natural", name: "Caltech-101", classes: 101, size: "object categories", shape: [224, 224, 3], url: "https://data.caltech.edu/records/mzrjq-6wc02" },
  { category: "Natural", name: "Food-101", classes: 101, size: "101k images", shape: [224, 224, 3], url: "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/" },
  { category: "Natural", name: "ImageNet Mini", classes: 1000, size: "subset", shape: [224, 224, 3], url: "https://www.image-net.org/" },
];

export const csvDatasets = [
  { category: "Classic: Classification", name: "Iris", target: "species", features: 4, samples: 150, url: "https://archive.ics.uci.edu/dataset/53/iris" },
  { category: "Classic: Classification", name: "Wine", target: "class", features: 13, samples: 178, url: "https://archive.ics.uci.edu/dataset/109/wine" },
  { category: "Classic: Classification", name: "Titanic", target: "Survived", features: 11, samples: 891, url: "https://www.kaggle.com/c/titanic" },
  { category: "Classic: Classification", name: "Digits (sklearn)", target: "digit", features: 64, samples: 1797, url: "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html" },

  { category: "Medical: Tabular", name: "Heart Disease (Cleveland)", target: "target", features: 13, samples: 303, url: "https://archive.ics.uci.edu/dataset/45/heart+disease" },
  { category: "Medical: Tabular", name: "Pima Diabetes", target: "Outcome", features: 8, samples: 768, url: "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database" },
  { category: "Medical: Tabular", name: "Breast Cancer Wisconsin", target: "diagnosis", features: 30, samples: 569, url: "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic" },
  { category: "Medical: Tabular", name: "Chronic Kidney Disease", target: "class", features: 24, samples: 400, url: "https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease" },

  { category: "Classic: Regression", name: "California Housing", target: "MedHouseVal", features: 8, samples: 20640, url: "https://www.kaggle.com/datasets/camnugent/california-housing-prices" },
  { category: "Classic: Regression", name: "Boston Housing (legacy)", target: "MEDV", features: 13, samples: 506, url: "https://www.kaggle.com/datasets/altavish/boston-housing-dataset" },
  { category: "Classic: Regression", name: "Auto MPG", target: "mpg", features: 7, samples: 398, url: "https://archive.ics.uci.edu/dataset/9/auto+mpg" },
];

export const augmentations = [
  "Resize", "RandomResizedCrop", "HorizontalFlip", "VerticalFlip",
  "Rotation", "ColorJitter", "GaussianBlur", "RandomErasing",
  "AutoAugment", "RandAugment", "MixUp", "CutMix", "Normalize",
];
