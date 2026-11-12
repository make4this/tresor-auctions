"""TreasuryDirect : résultats récents et calendrier des prochaines adjudications."""

from __future__ import annotations

import httpx
import pandas as pd

from tresor.http import fetch_json
from tresor.sources.raw import records_to_raw_frame

BASE_URL = "https://www.treasurydirect.gov/TA_WS/securities"
# Vérifié : malgré le paramètre days, l'API ne renvoie que les titres pas encore échus. Un bill
# à 4 semaines disparaît donc un mois après son émission : la collecte cumule les réponses.
RECENT_DAYS = 180

# Vérifié : type=Note et type=Bond ne renvoient ni les TIPS ni les FRN, il faut les demander.
SECURITY_TYPES = ("Bill", "Note", "Bond", "CMB", "TIPS", "FRN")

KEY_COLUMNS = ("cusip", "competitiveAuctionCloseDate")

UPCOMING_COLUMNS = (
    *KEY_COLUMNS,
    "announcementDate",
    "issueDate",
    "securityType",
    "securityTerm",
    "originalSecurityTerm",
    "reopening",
    "offeringAmount",
)

# Vérifié sur 99 enchères communes : ces champs valent exactement ceux de Fiscal Data.
RECENT_COLUMNS = (
    *UPCOMING_COLUMNS,
    "highYield",  # high_yield
    "lowYield",  # low_yield
    "medianYield",  # avg_med_yield
    "highRate",  # high_discnt_rate (bills)
    "investmentRate",  # high_investment_rate (bills)
    "lowRate",  # low_discnt_rate (bills)
    "medianRate",  # avg_med_discnt_rate (bills)
    "bidToCoverRatio",  # bid_to_cover_ratio
    "primaryDealersCompetitiveParAwarded",  # primary_dealer_accepted
    "directBiddersCompetitiveParAwarded",  # direct_bidder_accepted
    "indirectBiddersCompetitiveParAwarded",  # indirect_bidder_accepted
    "subtotalCompetitiveParRecognizedInsideOfOfferingAmount",  # comp_tendered
    "grandTotalParAwarded",  # total_accepted
    "grandTotalParRecognized",  # total_tendered
    "somaNoncompetitiveParAwarded",  # soma_accepted
    "inflationProtected",  # inflation_index_security
    "floatingRate",  # floating_rate
    "cashManagementBill",  # cash_management_bill_cmb
)


def fetch_recent(client: httpx.Client) -> pd.DataFrame:
    """Télécharge les résultats récents de tous les types de titres, sans doublon."""
    records = []
    for security_type in SECURITY_TYPES:
        params = {"format": "json", "type": security_type, "days": RECENT_DAYS}
        records.extend(fetch_json(f"{BASE_URL}/auctioned", params=params, client=client))
    return records_to_raw_frame(records).drop_duplicates(list(KEY_COLUMNS), ignore_index=True)


def fetch_upcoming(client: httpx.Client) -> pd.DataFrame:
    """Télécharge le calendrier ; le montant reste vide tant qu'il n'est pas annoncé."""
    records = fetch_json(f"{BASE_URL}/upcoming", params={"format": "json"}, client=client)
    return records_to_raw_frame(records)
