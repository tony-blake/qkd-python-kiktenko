# coding: utf-8

from argparse import ArgumentParser
import sys
from gettext import gettext as _


class ErrorPrintArgumentParser(ArgumentParser):
    def error(self, message):
        sys.stderr.write("{}\n".format(-1))
        self.print_usage(sys.stderr)
        self.exit(2, _('%s: error: %s\n') % (self.prog, message))