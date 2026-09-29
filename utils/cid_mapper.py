# utils/cid_mapper.py

import requests
import json
import os

CACHE_PATH = "data/processed/cid_cache.json"


class CIDMapper:

    def __init__(self):
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def save_cache(self):
        with open(CACHE_PATH, "w") as f:
            json.dump(self.cache, f, indent=2)

    def fetch_name(self, cid):

        cid = str(cid).lower().strip()
        cid_num = cid.replace("cid", "")

        if cid in self.cache:
            return self.cache[cid]

        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_num}/property/Title/JSON"
            r = requests.get(url, timeout=5)

            if r.status_code == 200:
                data = r.json()
                name = data["PropertyTable"]["Properties"][0]["Title"]

                self.cache[cid] = name.lower()
                self.save_cache()

                return name.lower()

        except Exception:
            pass

        self.cache[cid] = cid
        self.save_cache()
        return cid