from lightly.loss.vicreg_loss import VICRegLoss
from lightly.utils.debug import std_of_l2_normalized

from ..architectures.similarity import SimilarityEmbedding
from ..callbacks import SaveAugmentedSimilarityBatch
from .base import AmplfiModel


class SimilarityModel(AmplfiModel):
    """
    A LightningModule for training similarity embeddings

    Args:
        arch:
            A neural network architecture that maps waveforms
            to lower dimensional embedded space
    """

    def __init__(
        self,
        *args,
        arch: SimilarityEmbedding,
        similarity_loss: VICRegLoss,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # TODO: parmeterize cov, std, repr weights
        self.model = arch
        self.similarity_loss = similarity_loss

        # if checkpoint is not None, load in model weights;
        # checkpoint should only be specified in this way
        # if running trainer.test
        self.maybe_load_checkpoint(self.checkpoint)

    def forward(
        self,
        ref,
        aug,
    ):
        ref = self.model(ref)
        aug = self.model(aug)
        loss = self.similarity_loss(ref, aug)
        # loss, *elements = self.similarity_loss(ref, aug)
        return loss, (ref, aug)  # , elements

    def validation_step(self, batch, _):
        [ref, aug], asds, _ = batch
        loss, (ref, aug) = self((ref, asds), (aug, asds))
        ref_std = std_of_l2_normalized(ref)
        aug_std = std_of_l2_normalized(aug)
        self.log(
            "val_ref_std",
            ref_std,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "val_aug_std",
            aug_std,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "valid_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

        return loss

    def training_step(self, batch, _):
        # unpack batch - can ignore parameters
        [ref, aug], asds, _ = batch

        # pass reference and augmented data contexts
        # through embedding and calculate similarity loss
        loss, (ref, aug) = self((ref, asds), (aug, asds))
        ref_std = std_of_l2_normalized(ref)
        aug_std = std_of_l2_normalized(aug)
        self.log(
            "ref_std",
            ref_std,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "aug_std",
            aug_std,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()
        callbacks.append(SaveAugmentedSimilarityBatch())
        return callbacks
