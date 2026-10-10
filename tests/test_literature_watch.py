import importlib.util
import json
from pathlib import Path
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('literature_watch',REPO/'scripts/collect_eeg_literature.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BibliographyBoundaryTests(unittest.TestCase):
    def test_arxiv_keeps_publication_and_revision_distinct(self):
        data=b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2607.04139v2</id><title>A paper</title><published>2026-07-05</published><updated>2026-09-05</updated><summary>FULL ABSTRACT MUST NOT BE EXPORTED</summary><author><name>A Author</name></author></entry></feed>'
        result=module.parse_arxiv(data)
        self.assertNotEqual(result[0]['published'],result[0]['updated'])
        self.assertNotIn('FULL ABSTRACT',json.dumps(result))
        self.assertEqual(result[0]['url'],'https://arxiv.org/abs/2607.04139v2')
    def test_arxiv_error_feed_is_failure(self):
        with self.assertRaises(ValueError):module.parse_arxiv(b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/api/errors</id></entry></feed>')
    def test_crossref_index_date_is_not_publication(self):
        data=json.dumps({'status':'ok','message':{'items':[{'DOI':'10.1/test','title':['A paper'],'indexed':{'date-time':'2026-10-10'},'published-online':{'date-parts':[[2026,7,5]]},'abstract':'PRIVATE FULL ABSTRACT'}]}}).encode()
        record=module.parse_crossref(data)[0]
        self.assertEqual(record['published_online'],[[2026,7,5]])
        self.assertEqual(record['indexed'],'2026-10-10')
        self.assertNotIn('PRIVATE FULL ABSTRACT',json.dumps(record))
        self.assertIn('candidate',record['review_status'])
    def test_invalid_crossref_success_not_silently_empty(self):
        with self.assertRaises(ValueError):module.parse_crossref(b'{"status":"error","message":{}}')


if __name__=='__main__':unittest.main()
