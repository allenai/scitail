#!/usr/bin/env python
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

from scitail.commands import main  # pylint: disable=wrong-import-position

# Custom scitail modules for registering models and dataset readers
import scitail.models
import scitail.data.dataset_readers
import scitail.service.predictors

if __name__ == "__main__":
    main(prog="python -m scitail.run")
