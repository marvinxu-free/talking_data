# -*- coding: utf-8 -*-
from configparser import ConfigParser


class Config:
    def __init__(self, cfg_file, mode=0):
        self.mode = int(mode)
        self.cp = ConfigParser(allow_no_value=True)
        self.cp.read(cfg_file)

    def get_conf(self, domain, key):
        return self.cp.get(domain, key)

    def get_section(self, domain):
        return dict(self.cp.items(domain))

# cf = Config(0)
# cf = Config(1)
