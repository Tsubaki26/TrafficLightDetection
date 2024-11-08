import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, path="checkpoint_model.pth"):
        self.patience = patience  # ストップカウンタ。patience 回 loss の最小値を更新しなかったら学習を終了する。
        self.verbose = verbose  # 表示の有無
        self.path = path  # ベストモデル格納パス
        self.best_score = None  # ベストスコア
        self.early_stop = False  # ストップフラグ
        self.prev_best_score = np.Inf  # 前回のベストスコア
        self.counter = 0

    def __call__(self, loss, model):
        if self.best_score == None:
            # 1エポック目
            self.best_score = loss
            self.checkpoint(loss, model)
        elif loss > self.best_score:
            # ベストスコアを更新できなかった場合
            self.counter += 1
            if self.verbose:
                print(f"Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # ベストスコアを更新した場合
            self.counter = 0
            self.best_score = loss
            self.checkpoint(loss, model)

    def checkpoint(self, loss, model):
        if self.verbose:
            print(
                f"Loss decreased ({self.prev_best_score:.5f} --> {loss:.5f}). Saving model..."
            )
        # torch.save(model, self.path)
        model_scripted = torch.jit.script(model)
        model_scripted.save(self.path)
        self.prev_best_score = loss
