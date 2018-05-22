class Evaluator(object):

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.src_emb = trainer.src_emb
        self.tgt_emb = trainer.tgt_emb
        self.src_dic = trainer.src_dic
        self.tgt_dic = trainer.tgt_dic
        self.mapping = trainer.mapping
        self.discriminator = trainer.discriminator