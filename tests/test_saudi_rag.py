import unittest

from saudi_rag import Retriever, expand_query, normalize, tokenize
from saudi_rag.data import load_docs, load_queries
from saudi_rag.normalize import light_stem


class NormalizeTests(unittest.TestCase):
    def test_hamza_taa_marbuta_alef_maqsura(self):
        self.assertEqual(normalize("إلغاء الفاتورة على"), "الغاء الفاتوره علي")

    def test_diacritics_tatweel_digits(self):
        self.assertEqual(normalize("فَعّـــل ١٠٠"), "فعل 100")

    def test_light_stem_keeps_short_words(self):
        self.assertEqual(light_stem("الباقه"), "باق")
        self.assertEqual(light_stem("ال"), "ال")

    def test_tokenize_drops_stopwords(self):
        self.assertNotIn("في", tokenize("مشكلة في النت", stem=False))


class DialectTests(unittest.TestCase):
    def test_expansion_adds_msa(self):
        expanded = expand_query("النت مره ثقيل")
        self.assertIn("الانترنت", expanded)
        self.assertIn("بطيء", expanded)

    def test_conjunction_waw(self):
        self.assertIn("اريد", expand_query("وابي"))


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_docs()

    def test_top_hit(self):
        hits = Retriever(self.docs).search("ضاعت الشريحه وش اسوي", k=1)
        self.assertEqual(hits[0][0].id, "d04")

    def test_hybrid_beats_raw_on_dev_set(self):
        queries = load_queries()

        def r1(mode):
            r = Retriever(self.docs, mode=mode)
            return sum(r.search(q["query"], k=1)[0][0].id == q["gold"] for q in queries)

        self.assertGreater(r1("hybrid"), r1("raw"))

    def test_bad_mode(self):
        with self.assertRaises(ValueError):
            Retriever(self.docs, mode="magic")


if __name__ == "__main__":
    unittest.main()
