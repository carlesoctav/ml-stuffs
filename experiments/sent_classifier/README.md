usage: train.py [options] fit [-h] [-c CONFIG] [--print_config[=flags]] [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
                              [--trainer.accelerator.help CLASS_PATH_OR_NAME] [--trainer.accelerator ACCELERATOR]
                              [--trainer.strategy.help CLASS_PATH_OR_NAME] [--trainer.strategy STRATEGY] [--trainer.devices DEVICES]
                              [--trainer.num_nodes NUM_NODES] [--trainer.precision PRECISION]
                              [--trainer.logger.help CLASS_PATH_OR_NAME] [--trainer.logger LOGGER]
                              [--trainer.callbacks.help CLASS_PATH_OR_NAME] [--trainer.callbacks CALLBACKS]
                              [--trainer.fast_dev_run FAST_DEV_RUN] [--trainer.max_epochs MAX_EPOCHS]
                              [--trainer.min_epochs MIN_EPOCHS] [--trainer.max_steps MAX_STEPS] [--trainer.min_steps MIN_STEPS]
                              [--trainer.max_time MAX_TIME] [--trainer.limit_train_batches LIMIT_TRAIN_BATCHES]
                              [--trainer.limit_val_batches LIMIT_VAL_BATCHES] [--trainer.limit_test_batches LIMIT_TEST_BATCHES]
                              [--trainer.limit_predict_batches LIMIT_PREDICT_BATCHES] [--trainer.overfit_batches OVERFIT_BATCHES]
                              [--trainer.val_check_interval VAL_CHECK_INTERVAL]
                              [--trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH]
                              [--trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS] [--trainer.log_every_n_steps LOG_EVERY_N_STEPS]
                              [--trainer.enable_checkpointing {true,false,null}] [--trainer.enable_progress_bar {true,false,null}]
                              [--trainer.enable_model_summary {true,false,null}]
                              [--trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES]
                              [--trainer.gradient_clip_val GRADIENT_CLIP_VAL]
                              [--trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM] [--trainer.deterministic DETERMINISTIC]
                              [--trainer.benchmark {true,false,null}] [--trainer.inference_mode {true,false}]
                              [--trainer.use_distributed_sampler {true,false}] [--trainer.profiler.help CLASS_PATH_OR_NAME]
                              [--trainer.profiler PROFILER] [--trainer.detect_anomaly {true,false}]
                              [--trainer.barebones {true,false}] [--trainer.plugins.help CLASS_PATH_OR_NAME]
                              [--trainer.plugins PLUGINS] [--trainer.sync_batchnorm {true,false}]
                              [--trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS]
                              [--trainer.default_root_dir DEFAULT_ROOT_DIR] [--model CONFIG] [--model.model_name MODEL_NAME]
                              [--model.num_classes NUM_CLASSES] [--model.dropout DROPOUT] [--data CONFIG]
                              [--data.data_path DATA_PATH] [--data.model_name MODEL_NAME] [--data.batch_size BATCH_SIZE]
                              [--data.test_ratio TEST_RATIO] [--data.max_length MAX_LENGTH] [--optimizer CONFIG]
                              [--optimizer.lr.help CLASS_PATH_OR_NAME] [--optimizer.lr LR] [--optimizer.betas [ITEM,...]]
                              [--optimizer.eps EPS] [--optimizer.weight_decay WEIGHT_DECAY] [--optimizer.amsgrad {true,false}]
                              [--optimizer.foreach {true,false,null}] [--optimizer.maximize {true,false}]
                              [--optimizer.capturable {true,false}] [--optimizer.differentiable {true,false}]
                              [--optimizer.fused {true,false,null}] [--lr_scheduler CONFIG] --lr_scheduler.step_size STEP_SIZE
                              [--lr_scheduler.gamma GAMMA] [--lr_scheduler.last_epoch LAST_EPOCH]
                              [--lr_scheduler.verbose {true,false}] [--ckpt_path CKPT_PATH]

Runs the full optimization routine.

options:
  -h, --help            Show this help message and exit.
  -c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
  --print_config[=flags]
                        Print the configuration after applying all other arguments and exit. The optional flags customizes the output
                        and are one or more keywords separated by comma. The supported flags are: comments, skip_default, skip_null.
  --seed_everything SEED_EVERYTHING
                        Set to an int to run seed_everything with this value before classes instantiation.Set to True to use a random
                        seed. (type: Union[bool, int], default: True)

Customize every aspect of training via flags:
  --trainer CONFIG      Path to a configuration file.
  --trainer.accelerator.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Accelerator and exit.
  --trainer.accelerator ACCELERATOR
                        Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps", "auto") as well as
                        custom accelerator instances. (type: Union[str, Accelerator], default: auto, known subclasses:
                        lightning.pytorch.accelerators.CPUAccelerator, lightning.pytorch.accelerators.CUDAAccelerator,
                        lightning.pytorch.accelerators.MPSAccelerator, lightning.pytorch.accelerators.XLAAccelerator)
  --trainer.strategy.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Strategy and exit.
  --trainer.strategy STRATEGY
                        Supports different training strategies with aliases as well custom strategies. Default: ``"auto"``. (type:
                        Union[str, Strategy], default: auto, known subclasses: lightning.pytorch.strategies.DDPStrategy,
                        lightning.pytorch.strategies.DeepSpeedStrategy, lightning.pytorch.strategies.XLAStrategy,
                        lightning.pytorch.strategies.FSDPStrategy, lightning.pytorch.strategies.SingleDeviceStrategy,
                        lightning.pytorch.strategies.SingleDeviceXLAStrategy)
  --trainer.devices DEVICES, --trainer.devices+ DEVICES
                        The devices to use. Can be set to a positive number (int or str), a sequence of device indices (list or str),
                        the value ``-1`` to indicate all available devices should be used, or ``"auto"`` for automatic selection
                        based on the chosen accelerator. Default: ``"auto"``. (type: Union[List[int], str, int], default: auto)
  --trainer.num_nodes NUM_NODES
                        Number of GPU nodes for distributed training. Default: ``1``. (type: int, default: 1)
  --trainer.precision PRECISION
                        Double precision (64, '64' or '64-true'), full precision (32, '32' or '32-true'), 16bit mixed precision (16,
                        '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed'). Can be used on CPU, GPU, TPUs, HPUs or
                        IPUs. Default: ``'32-true'``. (type: Union[Literal[64, 32, 16], Literal['transformer-engine', 'transformer-
                        engine-float16', '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true'], Literal['64', '32',
                        '16', 'bf16'], null], default: null)
  --trainer.logger.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Logger and exit.
  --trainer.logger LOGGER, --trainer.logger+ LOGGER
                        Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses the default
                        ``TensorBoardLogger`` if it is installed, otherwise ``CSVLogger``. ``False`` will disable logging. If
                        multiple loggers are provided, local files (checkpoints, profiler traces, etc.) are saved in the ``log_dir``
                        of the first logger. Default: ``True``. (type: Union[Logger, Iterable[Logger], bool, null], default: null,
                        known subclasses: lightning.pytorch.loggers.logger.DummyLogger, lightning.pytorch.loggers.CometLogger,
                        lightning.pytorch.loggers.CSVLogger, lightning.pytorch.loggers.MLFlowLogger,
                        lightning.pytorch.loggers.NeptuneLogger, lightning.pytorch.loggers.TensorBoardLogger,
                        lightning.pytorch.loggers.WandbLogger)
  --trainer.callbacks.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Callback and exit.
  --trainer.callbacks CALLBACKS, --trainer.callbacks+ CALLBACKS
                        Add a callback or list of callbacks. Default: ``None``. (type: Union[List[Callback], Callback, null],
                        default: null, known subclasses: lightning.Callback, lightning.pytorch.callbacks.BatchSizeFinder,
                        lightning.pytorch.callbacks.Checkpoint, lightning.pytorch.callbacks.ModelCheckpoint,
                        lightning.pytorch.callbacks.OnExceptionCheckpoint, lightning.pytorch.callbacks.DeviceStatsMonitor,
                        lightning.pytorch.callbacks.EarlyStopping, lightning.pytorch.callbacks.BaseFinetuning,
                        lightning.pytorch.callbacks.BackboneFinetuning, lightning.pytorch.callbacks.GradientAccumulationScheduler,
                        lightning.pytorch.callbacks.LambdaCallback, lightning.pytorch.callbacks.LearningRateFinder,
                        lightning.pytorch.callbacks.LearningRateMonitor, lightning.pytorch.callbacks.ModelSummary,
                        lightning.pytorch.callbacks.RichModelSummary, lightning.pytorch.callbacks.BasePredictionWriter,
                        lightning.pytorch.callbacks.ProgressBar, lightning.pytorch.callbacks.RichProgressBar,
                        lightning.pytorch.callbacks.TQDMProgressBar, lightning.pytorch.callbacks.Timer,
                        lightning.pytorch.callbacks.ModelPruning, lightning.pytorch.callbacks.SpikeDetection,
                        lightning.pytorch.callbacks.StochasticWeightAveraging, lightning.pytorch.callbacks.ThroughputMonitor,
                        lightning.pytorch.cli.SaveConfigCallback)
  --trainer.fast_dev_run FAST_DEV_RUN
                        Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie:
                        a sort of unit test). Default: ``False``. (type: Union[int, bool], default: False)
  --trainer.max_epochs MAX_EPOCHS
                        Stop training once this number of epochs is reached. Disabled by default (None). If both max_epochs and
                        max_steps are not specified, defaults to ``max_epochs = 1000``. To enable infinite training, set ``max_epochs
                        = -1``. (type: Optional[int], default: null)
  --trainer.min_epochs MIN_EPOCHS
                        Force training for at least these many epochs. Disabled by default (None). (type: Optional[int], default:
                        null)
  --trainer.max_steps MAX_STEPS
                        Stop training after this number of steps. Disabled by default (-1). If ``max_steps = -1`` and ``max_epochs =
                        None``, will default to ``max_epochs = 1000``. To enable infinite training, set ``max_epochs`` to ``-1``.
                        (type: int, default: -1)
  --trainer.min_steps MIN_STEPS
                        Force training for at least these number of steps. Disabled by default (``None``). (type: Optional[int],
                        default: null)
  --trainer.max_time MAX_TIME
                        Stop training after this amount of time has passed. Disabled by default (``None``). The time duration can be
                        specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a :class:`datetime.timedelta`, or a
                        dictionary with keys that will be passed to :class:`datetime.timedelta`. (type: Union[str, timedelta,
                        Dict[str, int], null], default: null)
  --trainer.limit_train_batches LIMIT_TRAIN_BATCHES
                        How much of training dataset to check (float = fraction, int = num_batches). Default: ``1.0``. (type:
                        Union[int, float, null], default: null)
  --trainer.limit_val_batches LIMIT_VAL_BATCHES
                        How much of validation dataset to check (float = fraction, int = num_batches). Default: ``1.0``. (type:
                        Union[int, float, null], default: null)
  --trainer.limit_test_batches LIMIT_TEST_BATCHES
                        How much of test dataset to check (float = fraction, int = num_batches). Default: ``1.0``. (type: Union[int,
                        float, null], default: null)
  --trainer.limit_predict_batches LIMIT_PREDICT_BATCHES
                        How much of prediction dataset to check (float = fraction, int = num_batches). Default: ``1.0``. (type:
                        Union[int, float, null], default: null)
  --trainer.overfit_batches OVERFIT_BATCHES
                        Overfit a fraction of training/validation data (float) or a set number of batches (int). Default: ``0.0``.
                        (type: Union[int, float], default: 0.0)
  --trainer.val_check_interval VAL_CHECK_INTERVAL
                        How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check after a fraction of
                        the training epoch. Pass an ``int`` to check after a fixed number of training batches. An ``int`` value can
                        only be higher than the number of training batches when ``check_val_every_n_epoch=None``, which validates
                        after every ``N`` training batches across epochs or during iteration-based training. Default: ``1.0``. (type:
                        Union[int, float, null], default: null)
  --trainer.check_val_every_n_epoch CHECK_VAL_EVERY_N_EPOCH
                        Perform a validation loop every after every `N` training epochs. If ``None``, validation will be done solely
                        based on the number of training batches, requiring ``val_check_interval`` to be an integer value. Default:
                        ``1``. (type: Optional[int], default: 1)
  --trainer.num_sanity_val_steps NUM_SANITY_VAL_STEPS
                        Sanity check runs n validation batches before starting the training routine. Set it to `-1` to run all
                        batches in all validation dataloaders. Default: ``2``. (type: Optional[int], default: null)
  --trainer.log_every_n_steps LOG_EVERY_N_STEPS
                        How often to log within steps. Default: ``50``. (type: Optional[int], default: null)
  --trainer.enable_checkpointing {true,false,null}
                        If ``True``, enable checkpointing. It will configure a default ModelCheckpoint callback if there is no user-
                        defined ModelCheckpoint in :paramref:`~lightning.pytorch.trainer.trainer.Trainer.callbacks`. Default:
                        ``True``. (type: Optional[bool], default: null)
  --trainer.enable_progress_bar {true,false,null}
                        Whether to enable to progress bar by default. Default: ``True``. (type: Optional[bool], default: null)
  --trainer.enable_model_summary {true,false,null}
                        Whether to enable model summarization by default. Default: ``True``. (type: Optional[bool], default: null)
  --trainer.accumulate_grad_batches ACCUMULATE_GRAD_BATCHES
                        Accumulates gradients over k batches before stepping the optimizer. Default: 1. (type: int, default: 1)
  --trainer.gradient_clip_val GRADIENT_CLIP_VAL
                        The value at which to clip gradients. Passing ``gradient_clip_val=None`` disables gradient clipping. If using
                        Automatic Mixed Precision (AMP), the gradients will be unscaled before. Default: ``None``. (type: Union[int,
                        float, null], default: null)
  --trainer.gradient_clip_algorithm GRADIENT_CLIP_ALGORITHM
                        The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"`` to clip by value, and
                        ``gradient_clip_algorithm="norm"`` to clip by norm. By default it will be set to ``"norm"``. (type:
                        Optional[str], default: null)
  --trainer.deterministic DETERMINISTIC
                        If ``True``, sets whether PyTorch operations must use deterministic algorithms. Set to ``"warn"`` to use
                        deterministic algorithms whenever possible, throwing warnings on operations that don't support deterministic
                        mode. If not set, defaults to ``False``. Default: ``None``. (type: Union[bool, Literal['warn'], null],
                        default: null)
  --trainer.benchmark {true,false,null}
                        The value (``True`` or ``False``) to set ``torch.backends.cudnn.benchmark`` to. The value for
                        ``torch.backends.cudnn.benchmark`` set in the current session will be used (``False`` if not manually set).
                        If :paramref:`~lightning.pytorch.trainer.trainer.Trainer.deterministic` is set to ``True``, this will default
                        to ``False``. Override to manually set a different value. Default: ``None``. (type: Optional[bool], default:
                        null)
  --trainer.inference_mode {true,false}
                        Whether to use :func:`torch.inference_mode` or :func:`torch.no_grad` during evaluation
                        (``validate``/``test``/``predict``). (type: bool, default: True)
  --trainer.use_distributed_sampler {true,false}
                        Whether to wrap the DataLoader's sampler with :class:`torch.utils.data.DistributedSampler`. If not specified
                        this is toggled automatically for strategies that require it. By default, it will add ``shuffle=True`` for
                        the train sampler and ``shuffle=False`` for validation/test/predict samplers. If you want to disable this
                        logic, you can pass ``False`` and add your own distributed sampler in the dataloader hooks. If ``True`` and a
                        distributed sampler was already added, Lightning will not replace the existing one. For iterable-style
                        datasets, we don't do this automatically. (type: bool, default: True)
  --trainer.profiler.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Profiler and exit.
  --trainer.profiler PROFILER
                        To profile individual steps during training and assist in identifying bottlenecks. Default: ``None``. (type:
                        Union[Profiler, str, null], default: null, known subclasses: lightning.pytorch.profilers.AdvancedProfiler,
                        lightning.pytorch.profilers.PassThroughProfiler, lightning.pytorch.profilers.PyTorchProfiler,
                        lightning.pytorch.profilers.SimpleProfiler, lightning.pytorch.profilers.XLAProfiler)
  --trainer.detect_anomaly {true,false}
                        Enable anomaly detection for the autograd engine. Default: ``False``. (type: bool, default: False)
  --trainer.barebones {true,false}
                        Whether to run in "barebones mode", where all features that may impact raw speed are disabled. This is meant
                        for analyzing the Trainer overhead and is discouraged during regular training runs. The following features
                        are deactivated: :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_checkpointing`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.logger`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_progress_bar`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.log_every_n_steps`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.enable_model_summary`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.num_sanity_val_steps`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.fast_dev_run`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.detect_anomaly`,
                        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.profiler`,
                        :meth:`~lightning.pytorch.core.LightningModule.log`,
                        :meth:`~lightning.pytorch.core.LightningModule.log_dict`. (type: bool, default: False)
  --trainer.plugins.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of {Precision,ClusterEnvironment,CheckpointIO,LayerSync} and exit.
  --trainer.plugins PLUGINS, --trainer.plugins+ PLUGINS
                        Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins. Default:
                        ``None``. (type: Union[Precision, ClusterEnvironment, CheckpointIO, LayerSync, List[Union[Precision,
                        ClusterEnvironment, CheckpointIO, LayerSync]], null], default: null, known subclasses:
                        lightning.pytorch.plugins.Precision, lightning.pytorch.plugins.MixedPrecision,
                        lightning.pytorch.plugins.BitsandbytesPrecision, lightning.pytorch.plugins.DeepSpeedPrecision,
                        lightning.pytorch.plugins.DoublePrecision, lightning.pytorch.plugins.FSDPPrecision,
                        lightning.pytorch.plugins.HalfPrecision, lightning.pytorch.plugins.TransformerEnginePrecision,
                        lightning.pytorch.plugins.XLAPrecision, lightning.fabric.plugins.environments.KubeflowEnvironment,
                        lightning.fabric.plugins.environments.LightningEnvironment,
                        lightning.fabric.plugins.environments.LSFEnvironment, lightning.fabric.plugins.environments.MPIEnvironment,
                        lightning.fabric.plugins.environments.SLURMEnvironment,
                        lightning.fabric.plugins.environments.TorchElasticEnvironment,
                        lightning.fabric.plugins.environments.XLAEnvironment, lightning.fabric.plugins.TorchCheckpointIO,
                        lightning.fabric.plugins.XLACheckpointIO, lightning.pytorch.plugins.AsyncCheckpointIO,
                        lightning.pytorch.plugins.TorchSyncBatchNorm)
  --trainer.sync_batchnorm {true,false}
                        Synchronize batch norm layers between process groups/whole world. Default: ``False``. (type: bool, default:
                        False)
  --trainer.reload_dataloaders_every_n_epochs RELOAD_DATALOADERS_EVERY_N_EPOCHS
                        Set to a positive integer to reload dataloaders every n epochs. Default: ``0``. (type: int, default: 0)
  --trainer.default_root_dir DEFAULT_ROOT_DIR
                        Default path for logs and weights when no logger/ckpt_callback passed. Default: ``os.getcwd()``. Can be
                        remote file paths such as `s3://mybucket/path` or 'hdfs://path/' (type: Union[str, Path, null], default:
                        null)

<class '__main__.SentClassifierGivenHFModule'>:
  --model CONFIG        Path to a configuration file.
  --model.model_name MODEL_NAME
                        (type: Any, default: bert-base-uncased)
  --model.num_classes NUM_CLASSES
                        (type: Any, default: 2)
  --model.dropout DROPOUT
                        (type: Any, default: 0.2)

<class 'experiments.sent_classifier.dataset.DataFromJson'>:
  --data CONFIG         Path to a configuration file.
  --data.data_path DATA_PATH
                        (type: Any, default: ./data/raw/15239678.jsonl)
  --data.model_name MODEL_NAME
                        (type: Any, default: bert-base-uncased)
  --data.batch_size BATCH_SIZE
                        (type: Any, default: 16)
  --data.test_ratio TEST_RATIO
                        (type: Any, default: 0.2)
  --data.max_length MAX_LENGTH
                        (type: Any, default: 512)

Implements Adam algorithm:
  --optimizer CONFIG    Path to a configuration file.
  --optimizer.lr.help CLASS_PATH_OR_NAME
                        Show the help for the given subclass of Tensor and exit.
  --optimizer.lr LR     learning rate (default: 1e-3). A tensor LR is not yet supported for all our implementations. Please use a
                        float LR if you are not also specifying fused=True or capturable=True. (type: Union[float, Tensor], default:
                        0.001, known subclasses: torch.Tensor, torch.sparse.SparseSemiStructuredTensor, torch.nn.Parameter,
                        torch.nn.UninitializedParameter, torch.nn.UninitializedBuffer, torch.masked.MaskedTensor)
  --optimizer.betas [ITEM,...]
                        coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)) (type:
                        Tuple[float, float], default: (0.9, 0.999))
  --optimizer.eps EPS   term added to the denominator to improve numerical stability (default: 1e-8) (type: float, default: 1e-08)
  --optimizer.weight_decay WEIGHT_DECAY
                        weight decay (L2 penalty) (default: 0) (type: float, default: 0)
  --optimizer.amsgrad {true,false}
                        (default: False) (type: bool, default: False)
  --optimizer.foreach {true,false,null}
                        whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will
                        try to use foreach over the for-loop implementation on CUDA, since it is usually significantly more
                        performant. Note that the foreach implementation uses ~ sizeof(params) more peak memory than the for-loop
                        version due to the intermediates being a tensorlist vs just one tensor. If memory is prohibitive, batch fewer
                        parameters through the optimizer at a time or switch this flag to False (default: None) (type:
                        Optional[bool], default: null)
  --optimizer.maximize {true,false}
                        maximize the objective with respect to the params, instead of minimizing (default: False) (type: bool,
                        default: False)
  --optimizer.capturable {true,false}
                        whether this instance is safe to capture in a CUDA graph. Passing True can impair ungraphed performance, so
                        if you don't intend to graph capture this instance, leave it False (default: False) (type: bool, default:
                        False)
  --optimizer.differentiable {true,false}
                        whether autograd should occur through the optimizer step in training. Otherwise, the step() function runs in
                        a torch.no_grad() context. Setting to True can impair performance, so leave it False if you don't intend to
                        run autograd through this instance (default: False) (type: bool, default: False)
  --optimizer.fused {true,false,null}
                        whether the fused implementation (CUDA only) is used. Currently, `torch.float64`, `torch.float32`,
                        `torch.float16`, and `torch.bfloat16` are supported. (default: None) (type: Optional[bool], default: null)

Decays the learning rate of each parameter group by gamma every:
  --lr_scheduler CONFIG
                        Path to a configuration file.
  --lr_scheduler.step_size STEP_SIZE
                        Period of learning rate decay. (required, type: int)
  --lr_scheduler.gamma GAMMA
                        Multiplicative factor of learning rate decay. Default: 0.1. (type: float, default: 0.1)
  --lr_scheduler.last_epoch LAST_EPOCH
                        The index of last epoch. Default: -1. (type: int, default: -1)
  --lr_scheduler.verbose {true,false}
                        If ``True``, prints a message to stdout for each update. Default: ``False``. .. deprecated:: 2.2 ``verbose``
                        is deprecated. Please use ``get_last_lr()`` to access the learning rate. (type: bool, default: deprecated)

Runs the full optimization routine:
  --ckpt_path CKPT_PATH
                        Path/URL of the checkpoint from which training is resumed. Could also be one of two special keywords
                        ``"last"`` and ``"hpc"``. If there is no checkpoint file at the path, an exception is raised. (type:
                        Union[str, Path, null], default: null)
(     whether to use the AMSGrad variant of this algorithm from the paper `On the Convergence of Adam and Beyond`_
                   
