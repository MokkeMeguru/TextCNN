import sentencepiece as spm


class CustomTextEncoder():
    def __init__(self, sp: spm.SentencePieceProcessor):
        "docstring"
        self.sp = sp
        self.vocab_size = self.sp.GetPieceSize()
        assert self.sp.unk_id(
        ) != -1, 'this sentencepiece model has no vocab means "unknown"'

    def encode(self, x):
        return self.sp.EncodeAsIds(x)

    def encode_words(self, x):
        return self.sp.EncodeAsPieces(x)

    def decode(self, ids):
        return self.sp.DecodeIds(ids)

    def vocab_size(self):
        return self.sp.GetPieceSize()

    def save_to_file(self, filename_prefix):
        raise NotImplementedError

    def load_from_file(cls, filename_prefix):
        raise NotImplementedError

    def encode_word(self, word):
        return self.sp.PieceToId(word)

    def decode_id(self, i):
        return self.sp.IdToPiece(i)
