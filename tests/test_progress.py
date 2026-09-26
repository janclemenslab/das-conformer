from types import SimpleNamespace

from lightning.pytorch.callbacks import TQDMProgressBar

from das import api
from das.progress import TrainingLogProgress


def test_training_log_progress_reports_batches_without_terminal_redraw(monkeypatch, capsys):
    monkeypatch.setattr("das.progress.time.monotonic", lambda: 1.0)
    trainer = SimpleNamespace(current_epoch=1, max_epochs=3, num_training_batches=10)
    progress = TrainingLogProgress()
    progress.on_train_epoch_start(trainer, None)
    for batch_idx in range(10):
        progress.on_train_batch_end(trainer, None, None, None, batch_idx)

    assert capsys.readouterr().out.splitlines() == [
        "Epoch 2/3: [##------------------] 1/10 batches (10%)",
        "Epoch 2/3: [####################] 10/10 batches (100%)",
    ]


def test_training_uses_terminal_bar_or_log_progress():
    terminal = api._training_callbacks(None, stop_event=None, verbose=True, emit_epoch_logs=False)
    gui = api._training_callbacks(None, stop_event=None, verbose=True, emit_epoch_logs=True)

    assert any(isinstance(callback, TQDMProgressBar) for callback in terminal)
    assert any(isinstance(callback, TrainingLogProgress) for callback in gui)
    assert not any(isinstance(callback, TQDMProgressBar) for callback in gui)
