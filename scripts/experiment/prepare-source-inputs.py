"""Prepare generation and DCLM name inputs from a supplied gender-first JSON."""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def write_name_inputs(data, country_names, output_dir, max_names_per_country=30, write_json=True):
    """Convert country -> gender -> names to the generator's gender-first JSON."""
    by_gender = {'M': {}, 'F': {}}
    rows = []
    for code, groups in data.items():
        seen = set()
        for gender in by_gender:
            names = groups.get(gender, [])
            if not names:
                raise ValueError(f'Missing {gender} names for {code}.')
            by_gender[gender][code] = names
            for name in names[:max_names_per_country]:
                if name not in seen:
                    rows.append({'country': country_names[code], 'name': name, 'sex': gender})
                    seen.add(name)
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_json:
        (output_dir / 'fb_names.json').write_text(json.dumps(by_gender, ensure_ascii=False, indent=2) + '\n')
    with (output_dir / 'names.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['country', 'name', 'sex'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fb-names', type=Path, required=True, help='JSON with M/F groups keyed by ISO country code.')
    parser.add_argument('--max-names-per-country', type=int, default=30, help='Names per gender in names.csv; match generator settings.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'external/data')
    args = parser.parse_args()
    if args.max_names_per_country < 1:
        parser.error('--max-names-per-country must be positive.')
    import pycountry
    source = json.loads(args.fb_names.read_text(encoding='utf-8'))
    codes = sorted(set(source.get('M', {})) | set(source.get('F', {})))
    countries = {}
    for code in codes:
        country = pycountry.countries.get(alpha_2=code)
        if country is None:
            parser.error(f'Unknown ISO country code: {code}')
        countries[code] = country.name
    if not countries:
        parser.error('The JSON contains no country names.')
    data = {code: {gender: source.get(gender, {}).get(code, []) for gender in ['M', 'F']}
            for code in countries}
    write_name_inputs(data, countries, args.output_dir, args.max_names_per_country,
                      write_json=args.fb_names.resolve() != (args.output_dir / 'fb_names.json').resolve())
    print(f'Source inputs prepared in {args.output_dir}')


if __name__ == '__main__':
    main()
