import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from urllib.parse import urljoin
    import re

    import requests


    BASE_URL = "https://ftp.nyse.com/cta_symbol_files/"
    TARGET_YYYYMM = "202509"
    OUT_DIR = Path(f"data/cta_symbol_files_{TARGET_YYYYMM}")


    resp = requests.get(BASE_URL, timeout=30)
    resp.raise_for_status()

    filenames = sorted(set(re.findall(
        rf'CTA\.Symbol\.File\.{TARGET_YYYYMM}\d{{2}}\.csv',
        resp.text
    )))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in filenames:
        url = urljoin(BASE_URL, name)
        path = OUT_DIR / name

        if path.exists():
            print(f"skip   {name}")
            continue

        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"saved  {name}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
