import os
import re


def _get_version():
    filename = os.path.join(os.path.dirname(__file__), 'CHANGELOG.md')
    with open(filename, 'rt') as file:
        pat = r"""
            (?P<version>\d+\.\d+)         # minimum 'N.N'
            (?P<extraversion>(?:\.\d+)*)  # any number of extra '.N' segments
            (?:
                (?P<prerel>[abc]|rc)      # 'a' = alpha, 'b' = beta
                                          # 'c' or 'rc' = release candidate
                (?P<prerelversion>\d+(?:\.\d+)*)
            )?
            (?P<postdev>(\.post(?P<post>\d+))?(\.dev(?P<dev>\d+))?)?
        """
        for line in file:
            match = re.search(pat, line, re.VERBOSE)
            if match:
                return match.group()
    raise ValueError("Can't get version")


__version__ = _get_version()
