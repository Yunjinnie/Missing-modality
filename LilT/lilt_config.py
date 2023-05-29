from sacred import Experiment

ex = Experiment("LilT")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "mppd": 0,
        "vqa": 0,
        "nlvr2": 0,
        "irtr": 0,
        "mmimdb": 0,
        "hatememes": 0,
        "food101": 0,        
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "LilT"
    seed = 0
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.

    # eval config (for bash execution)
    test_ratio = None
    test_type = None
    test_exp_name = None
    
    # fix backbone model (ViLT) weights
    fix_model = False
    
    # missing modality config
    missing_ratio = {'train': 0.0, 'val': 0.0, 'test': 0.0} #
    missing_type = {'train': 'both', 'val': 'both', 'test': 'both'} # ['text', 'image', 'both'] in VL taskss
    both_ratio = 0.5   # missing both ratio
    missing_table_root = './datasets/missing_tables/' #?
    simulate_missing = False


    hidden_size = 768


    # LilT setting
    # _BERT_CONFIG_MAP = {
    #     "large": "princeton-nlp/unsup-simcse-bert-large-uncased",
    #     "base": "princeton-nlp/unsup-simcse-bert-base-uncased",
    #     "base_mlm": "bert-base-uncased",
    #     "small": "prajjwal1/bert-small",
    #     "tiny": "prajjwal1/bert-tiny",
    #     "base_multilingual": "bert-base-multilingual-cased",
    # }
    vision_encoder = "base"
    text_encoder = "base"
    pretrained_text = True
    pretrained_vision= True
    freeze_text_encoder =  True
    freeze_vision_encoder=  True
    freeze_proj=  False


    unlock_layernorm=  False
    limit_num_samples=  False
    unlock_dense=  False
    unlock_attn=  False
    unlock_random=  False
    bitfit=  False ##

    add_adapter=  False # unfreeze last layer
    adapter_append=  False
    fp16=  False

    embed_dim= 384
    image_res= 384 #256
    always_freeze = {"visual_encoder": [],"text_encoder": []}
    conventional_adapter={ "insert": True ,"reduction_factor": 4}

    # Image setting
    train_transform_keys = ["pixelbert"]
    val_transform_keys = ["pixelbert"]
    image_size = 384
    max_image_len = -1
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqav2_label_size = 3129
    max_text_len = 40
    tokenizer = "large"
    vocab_size = 30522
    whole_word_masking = False
    mlm_prob = 0.15
    draw_false_text = 0

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    max_steps = 25000
    warmup_steps = 2500
    end_lr = 0
    lr_mult = 1  # multiply lr for downstream heads

    # Downstream Setting
    get_recall_metric = False
    mmimdb_class_num = 23
    hatememes_class_num = 2
    food101_class_num = 101    

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    finetune_first = False


    # below params varies with the environment
    data_root = "datasets/mmimdb"
    log_dir = "result_LilT"
    per_gpu_batchsize = 4  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 3 #
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16
    wandb_name = "missing" ## edit
    wandb_project = "task_finetune_mmimdb"


# Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
@ex.named_config
def task_mlm_itm():
    exp_name = "mlm_itm"
    datasets = ["coco", "vg", "sbu", "gcc"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096
    max_epoch = 10
    max_image_len = 200

    
@ex.named_config
def task_finetune_mmimdb():
    exp_name = "finetune_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
#     loss_names = _loss_names({"mmimdb": 1, "prompt": -0.5})
    batch_size = 256
    max_epoch = 25
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 2e-2
#     optim_type = "adam"
    max_text_len = 512

@ex.named_config
def task_lilt_mmimdb():
    exp_name = "finetune_lilt_mmimdb"
    datasets = ["mmimdb"]
    loss_names = _loss_names({"mmimdb": 1})
#     loss_names = _loss_names({"mmimdb": 1, "prompt": -0.5})
    batch_size = 256
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.1
    draw_false_image = 0
    learning_rate = 1e-2
    val_check_interval = 0.2
    weight_decay = 5e-2
#     optim_type = "adam"
    max_text_len = 512


@ex.named_config
def step25k():
    max_epoch = 100
    max_steps = 25000


@ex.named_config
def step50k():
    max_epoch = 100
    max_steps = 50000


@ex.named_config
def step100k():
    max_epoch = 100
    max_steps = 100000


@ex.named_config
def step200k():
    max_epoch = 200
    max_steps = 200000

