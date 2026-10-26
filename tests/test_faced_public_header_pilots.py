import importlib.util
from pathlib import Path
import unittest

REPO=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('faced_pilot',REPO/'scripts/check_faced_public_headers.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class DestinationTests(unittest.TestCase):
    def test_only_expected_public_broker_host_is_allowed(self):
        module.validate_redirect('https://nemar.s3.us-east-2.amazonaws.com/public?temporary=not-retained')
        for url in ('http://nemar.s3.us-east-2.amazonaws.com/a','https://nemar.s3.us-east-2.amazonaws.com.evil.test/a','https://user:password@nemar.s3.us-east-2.amazonaws.com/a','https://example.com/a','https://nemar.s3.us-east-2.amazonaws.com/a#fragment'):
            with self.assertRaises(ValueError):module.validate_redirect(url)


if __name__=='__main__':unittest.main()
