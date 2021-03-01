from example import *
from data import *
from tqdm import tqdm
from random import randint
import cv2


class CFG:
    root_path = "./"
    origin_path = ""
    model_path = "./"
    log_path = "./"
    work_dir = './new_train'

    model_name = "BaseModel"

    image_size = 256 # [256, 512, 1024, 'original']
    use_class14 = False

    n_folds = 5
    seed = 42
    val_fold = 0

    fusion_mode = "nms"  # choice of [nms, soft_nms, wbf]
    box_count = 2
    reset = True


CFG.debug = False

CFG.batch_size = 8
CFG.num_epochs = 32
CFG.learning_rate = 1e-3

CFG.pretrained_path = None

if CFG.image_size == 512:
    CFG.image_path = os.path.join(CFG.root_path, "vinbigdata")
elif CFG.image_size == 256:
    CFG.image_path = CFG.root_path
elif CFG.image_size == 'original':
    CFG.image_path = os.path.join('original-data', 'vinbigdata')
else:
    CFG.image_path = os.path.join(CFG.root_path, f"vinbigdata-chest-xray-resized-png-{CFG.image_size}x{CFG.image_size}")

CFG.device = get_device()
CFG.workers = 16

CFG.fusion_mode = "nms"

# set seed
seed_everything(CFG.seed)

# load data
if CFG.reset:
    meta_df = load_all_data(CFG)
    # get fold
    df_folds = get_folds(meta_df, CFG)
    df_5 = df_folds[df_folds['fold'] != 4].reset_index().rename(columns={'index': 'image_id'})
    df_5.to_csv('df_5.csv', index=True, columns=['image_id', 'bbox_count', 'fold'])
    all_data_df = load_all_data(CFG)
    all_data_df.to_csv('all_data.csv', index=True,
                       columns=['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max',
                                'x_min_resized', 'y_min_resized', 'x_max_resized', 'y_max_resized', 'dim0', 'dim1'])
    normal_data_df = load_normal_data(CFG)
    normal_data_df.to_csv('normal_data_df.csv', index=True, columns=['image_id', 'class_name', 'class_id', 'rad_id',
                                                                     'x_min', 'y_min', 'x_max', 'y_max', 'dim0',
                                                                     'dim1'])
else:
    all_data_df = pd.read_csv('all_data.csv')
    normal_data_df = pd.read_csv('normal_data_df.csv')
    df_5 = pd.read_csv('df_5.csv')

dataset = CustomDataset(all_data_df[all_data_df['class_id'] != 14], df_5, CFG, normal=False)
normal_dataset = NormalDataset(normal_data_df, CFG)

import pathlib
pathlib.Path(f'{CFG.work_dir}_{CFG.image_size}').mkdir(parents=True, exist_ok=True)
i = 0
df = pd.DataFrame(columns=['image_id', 'class_id', 'x_min', 'y_min', 'x_max', 'y_max',
                           'x_min_resized', 'y_min_resized', 'x_max_resized', 'y_max_resized', 'width', 'height'])
df = df.append(df_5)
normal_image = None
for image, box, original_box in tqdm(dataset):
    if i == 3 * 10606:
        break
    if i % 3 == 0:
        if i != 0:
            if CFG.image_size == 'original':
                cv2.imwrite(os.path.join(f'{CFG.work_dir}_{CFG.image_size}', image_id + 'abcde' + '.jpg'), normal_image)
            else:
                cv2.imwrite(os.path.join(f'{CFG.work_dir}_{CFG.image_size}', image_id + 'abcde' + '.png'), normal_image)
        normal_image, image_id = next(iter(normal_dataset))
    roi = image[box[1]:box[3], box[0]:box[2]].copy()
    width = box[3] - box[1]
    height = box[2] - box[0]
    try:
        temp_height = randint(0, normal_image.shape[1] - height)
        temp_width = randint(0, normal_image.shape[0] - width)
    except:
        continue
    normal_image[temp_width:temp_width + width, temp_height:temp_height + height] = roi
    if CFG.image_size == 'original':
        df = df.append({
            'image_id': image_id + 'abcde',
            'class_id': box[4],
            'x_min': temp_width,
            'y_min': temp_height,
            'x_max': (temp_width + width),
            'y_max': (temp_height + height),
        }, ignore_index=True)
    else:
        df = df.append({
            'image_id': image_id + 'abcde',
            'class_id': box[4],
            'x_min': temp_width * (original_box[5] / CFG.image_size),
            'y_min': temp_height * (original_box[4] / CFG.image_size),
            'x_max': (temp_width + width) * (original_box[5] / CFG.image_size),
            'y_max': (temp_height + height) * (original_box[4] / CFG.image_size),
            'x_min_resized': temp_width,
            'y_min_resized': temp_height,
            'x_max_resized': temp_width + width,
            'y_max_resized': temp_height + height
        }, ignore_index=True)
    i += 1

df.to_csv(f'result_{CFG.image_size}.csv')
