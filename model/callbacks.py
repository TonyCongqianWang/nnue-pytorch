import lightning as L

from .lightning_module import NNUE


class WeightClippingCallback(L.Callback):
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch,
        batch_idx: int,
    ) -> None:
        assert isinstance(pl_module, NNUE)
        pl_module.model.clip_weights()
        if batch_idx == 0:
            pl_module.model.clip_threat_weights()

class PhaseTrainingScheduleCallback(L.Callback):
    def __init__(self, schedule: dict):
        self.schedule = schedule

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        
        if epoch in self.schedule:
            phase_config = self.schedule[epoch]
            nnue_model = pl_module.model 
            
            print(f"\n--- [Epoch {epoch}] Entering new training phase ---")
            
            # 1. Apply Fake Quantization
            if "quantize" in phase_config:
                target = phase_config["quantize"]
                print(f"-> Applying fake quantization to: {target}")
                nnue_model.replace_with_quantized_weights(target)
                
            # 2. Update requires_grad flags
            if "learnable" in phase_config:
                print(f"-> Updating learnable modules: {phase_config['learnable']}")
                nnue_model.set_learnable_modules(phase_config["learnable"])
                
            # 3. Reset Optimizer and Learning Rate Scheduler
            if phase_config.get("reset_optimizer", False):
                print("-> Resetting optimizer state to prevent retroactive weight changes.")
                for optimizer in trainer.optimizers:
                    optimizer.state.clear()
                    if hasattr(optimizer, '_reset_state_dict'):
                         optimizer._reset_state_dict()
            
            if phase_config.get("reset_scheduler", False):
                print("-> Rewinding LR Scheduler back to initial epoch 0.")
                for config in trainer.lr_scheduler_configs:
                    scheduler = config.scheduler
                    
                    scheduler.last_epoch = 0
                    if hasattr(scheduler, "_step_count"):
                        scheduler._step_count = 0
                        
                    if hasattr(scheduler, "base_lrs"):
                        for optimizer in trainer.optimizers:
                            for param_group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                                param_group["lr"] = base_lr
                            
            # 4. Toggle Forward Pass Short-Circuiting (defaults to False)
            is_psqt_only = phase_config.get("train_psqt_only", False)
            nnue_model.train_psqt_only = is_psqt_only
            print(f"-> Short-circuiting L1/LayerStacks: {'ON' if is_psqt_only else 'OFF'}")
                    
            print("---------------------------------------------------\n")