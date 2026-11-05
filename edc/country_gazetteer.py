"""Self-contained world country gazetteer.

This list is derived from the ISO 3166 standard set of sovereign countries plus
widely used common/short names and historical variants. It is deliberately an
EXTERNAL, general-knowledge resource and is NOT built from any WebNLG/eval
reference — using a general country list to decide whether an object is a
country avoids fitting rules to the test set.

Used by the output post-processor to repair the common `location -> country`
confusion: an object that is a recognised country name almost never denotes a
generic "location" relation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Set


# ISO 3166 short names (sovereign states). Kept lowercase for matching.
_ISO_COUNTRIES = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
    "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada",
    "Central African Republic", "Chad", "Chile", "China", "Colombia",
    "Comoros", "Congo", "Costa Rica", "Croatia", "Cuba", "Cyprus",
    "Czech Republic", "Czechia", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji",
    "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana",
    "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran",
    "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan",
    "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kosovo", "Kuwait",
    "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
    "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania",
    "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia", "Nauru",
    "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria",
    "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru",
    "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia",
    "Rwanda", "Saint Kitts and Nevis", "Saint Lucia",
    "Saint Vincent and the Grenadines", "Samoa", "San Marino",
    "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
    "Somalia", "South Africa", "South Korea", "South Sudan", "Spain",
    "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo",
    "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan",
    "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
]

# Common short forms / historical / alternate spellings widely used in text.
_ALIASES = [
    "USA", "U.S.A.", "US", "U.S.", "United States of America", "America",
    "UK", "U.K.", "Great Britain", "Britain", "England", "Scotland", "Wales",
    "Northern Ireland", "Republic of Ireland", "Holland", "Russian Federation",
    "South Korea", "Republic of Korea", "North Korea", "Burma",
    "Ivory Coast", "Cote d'Ivoire", "Cape Verde", "Swaziland", "Macedonia",
    "UAE", "Vatican", "Palestinian territories", "Democratic Republic of the Congo",
    "DR Congo", "Republic of the Congo",
]


def _norm(value: str) -> str:
    value = str(value).strip().strip('"').strip("'")
    value = value.replace("_", " ")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value)
    return value.lower().strip()


COUNTRY_KEYS: Set[str] = {_norm(name) for name in _ISO_COUNTRIES + _ALIASES}


def is_country(value: str) -> bool:
    """Return True if the (surface) value names a recognised country."""
    return _norm(value) in COUNTRY_KEYS
