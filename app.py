
import json
import os
import re
import uuid
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import streamlit as st

try:
    from groq import Groq
except ImportError:
    Groq = None


st.set_page_config(
    page_title="PayPilot | AI Money Agent",
    page_icon="💸",
    layout="wide",
)

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

.stApp {
    background: #0b1020;
    color: #edf2ff;
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    color: #f3f6ff !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #11182b;
}
[data-testid="stMetric"] {
    background: #151e33;
    border: 1px solid #293652;
    border-radius: 15px;
    padding: 17px;
}
[data-testid="stMetricLabel"] {
    color: #aab7d1 !important;
}
[data-testid="stMetricValue"] {
    color: #f6f8ff !important;
}
div.stButton > button {
    border-radius: 10px;
    min-height: 42px;
    font-weight: 600;
}
.hero {
    background: linear-gradient(120deg, #182c53, #172044 60%, #291b4b);
    border: 1px solid #33466e;
    border-radius: 22px;
    padding: 28px;
    margin: 8px 0 22px 0;
}
.eyebrow {
    color: #91a8ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
}
.panel {
    background: #131c30;
    border: 1px solid #293652;
    border-radius: 16px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# ---------- Session state ----------
if "transactions" not in st.session_state:
    st.session_state.transactions = [
        {
            "id": "DEMO-1042",
            "recipient": "Jordan Lee",
            "amount": 28.50,
            "purpose": "Team lunch",
            "status": "Sample — not a real payment",
            "created": "Today",
        },
        {
            "id": "DEMO-1041",
            "recipient": "Cloud Studio",
            "amount": 14.00,
            "purpose": "Design subscription",
            "status": "Sample — not a real payment",
            "created": "Yesterday",
        },
    ]

if "pending" not in st.session_state:
    st.session_state.pending = None


# ---------- AI and parsing ----------
def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return os.getenv(name, default)


def local_parse(text):
    amount_match = re.search(
        r"\$\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)"
        r"|(\d+(?:\.\d{1,2})?)\s*(?:USD|dollars?)\b",
        text,
        re.I,
    )

    if not amount_match:
        raise ValueError("Include an amount, such as $35.")

    raw = next(g for g in amount_match.groups() if g)
    amount = Decimal(raw.replace(",", ""))

    if not amount.is_finite() or not 0 < amount <= 10000:
        raise ValueError("The amount must be between $0.01 and $10,000.")

    recipient_match = re.search(
        r"\bto\s+(.+?)(?=\s+for\b|[,.!?]|$)",
        text,
        re.I,
    )
    recipient = recipient_match.group(1).strip() if recipient_match else ""

    purpose_match = re.search(
        r"\bfor\s+(.+?)(?:[.!?]|$)",
        text,
        re.I,
    )
    purpose = purpose_match.group(1).strip() if purpose_match else "General payment"

    if not recipient:
        raise ValueError("Tell me who the payment is for, e.g. 'to Alex'.")

    return {
        "recipient": recipient[:80],
        "amount": float(amount.quantize(Decimal("0.01"))),
        "purpose": purpose[:120],
    }



def make_plan(text):
    api_key = get_secret("GROQ_API_KEY")
    model = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

    if not api_key or Groq is None:
        data = local_parse(text)
        data["engine"] = "Local parser (demo mode)"
        return data

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PayPilot, a payment planning assistant. "
                    "Extract payment requests into JSON with fields "
                    "recipient (string), amount (number or null), and "
                    "purpose (string). Never execute payments or claim "
                    "that money has moved. Never invent a missing recipient "
                    "or amount. If details are unclear, use an empty "
                    "recipient or null amount. Amounts must be positive."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    import json

    data = json.loads(response.choices[0].message.content)

    return {
        "recipient": str(data.get("recipient") or "").strip(),
        "amount": data.get("amount"),
        "purpose": str(data.get("purpose") or "General payment")[:120],
        "engine": "Groq AI",
    }


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("# 💸 PayPilot")
    st.caption("YOUR AI MONEY AGENT")
    st.divider()

    page = st.radio(
        "WORKSPACE",
        ["Overview", "AI Assistant", "Transactions", "About"],
    )

    st.divider()
    st.markdown("### Environment")
    st.success("Demo mode active")

    if get_secret("GROQ_API_KEY") and Groq:
        st.caption("🟢 Groq AI connected")
    else:
        st.caption("🟡 Local parser active")

    st.warning("No real payments are sent.")

    if st.button("Reset demo history"):
        st.session_state.transactions = []
        st.session_state.pending = None
        st.rerun()


# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="eyebrow">INTELLIGENT PAYMENTS · BUILT FOR PEOPLE</div>
    <h1>Your money. Your intent.<br>One intelligent assistant.</h1>
    <p>Describe a payment. Review the plan. Stay in control.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Overview ----------
if page == "Overview":
    st.subheader("Your financial command center")
    st.caption("Illustrative data only — not a connected PayPal account.")

    transactions = st.session_state.transactions
    total = sum(float(t["amount"]) for t in transactions)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tracked activity", f"${total:,.2f}")
    c2.metric("Activity records", len(transactions))
    c3.metric(
        "Payment plan",
        "Awaiting review" if st.session_state.pending else "Ready",
    )

    st.markdown("### Start with a request")
    st.write("Try: **Prepare a $35 payment to Alex for dinner.**")

    if st.button("Create payment plan →", type="primary"):
        st.session_state.page = "AI Assistant"
        st.rerun()

    st.markdown("### Recent activity")
    for t in transactions[:5]:
        with st.container(border=True):
            a, b, c = st.columns([2, 1, 2])
            a.markdown(f"**{t['recipient']}**")
            b.markdown(f"**${t['amount']:,.2f}**")
            c.caption(t["status"])


# ---------- AI Assistant ----------
elif page == "AI Assistant":
    st.subheader("✨ AI payment assistant")
    st.write("Describe the payment you'd like to prepare.")

    with st.form("payment_request"):
        request = st.text_area(
            "Payment request",
            placeholder="Prepare a $35 payment to Alex for dinner.",
        )
        submitted = st.form_submit_button(
            "Generate payment plan",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not request.strip():
            st.warning("Enter a payment request.")
        else:
            try:
                with st.spinner("Preparing your plan..."):
                    plan = make_plan(request)

                recipient = plan.get("recipient", "")
                amount = plan.get("amount")
                purpose = plan.get("purpose", "General payment")

                if not recipient:
                    st.error("The recipient is missing. Please try again.")
                elif isinstance(amount, bool) or amount is None:
                    st.error("The amount is missing. Please try again.")
                else:
                    amount = Decimal(str(amount))

                    if not amount.is_finite() or not 0 < amount <= 10000:
                        st.error("The amount must be between $0.01 and $10,000.")
                    else:
                        st.session_state.pending = {
                            "id": uuid.uuid4().hex[:8].upper(),
                            "recipient": recipient[:80],
                            "amount": float(amount.quantize(Decimal("0.01"))),
                            "purpose": purpose[:120],
                            "engine": plan["engine"],
                        }
                        st.rerun()

            except Exception as e:
                st.error(f"Could not prepare the plan: {e}")

    plan = st.session_state.pending

    if plan:
        st.markdown("### Review your payment")

        with st.container(border=True):
            st.caption(f"PLAN ID · {plan['id']}")
            st.subheader(f"${plan['amount']:,.2f}")
            st.write(f"**Recipient:** {plan['recipient']}")
            st.write(f"**Purpose:** {plan['purpose']}")
            st.caption(f"Prepared using: {plan['engine']}")

            st.warning(
                "SIMULATED PAYMENT — this action does not contact PayPal "
                "or transfer money."
            )

            confirmed = st.checkbox(
                "I reviewed the recipient, amount, and purpose."
            )

            col1, col2 = st.columns(2)

            if col1.button(
                "Confirm demo action",
                type="primary",
                use_container_width=True,
            ):
                if not confirmed:
                    st.error("Please review and check the confirmation box.")
                else:
                    st.session_state.transactions.insert(
                        0,
                        {
                            "id": "SIM-" + uuid.uuid4().hex[:8].upper(),
                            "recipient": plan["recipient"],
                            "amount": plan["amount"],
                            "purpose": plan["purpose"],
                            "status": "Simulated — not paid",
                            "created": datetime.now(timezone.utc).strftime(
                                "%Y-%m-%d %H:%M UTC"
                            ),
                        },
                    )
                    st.session_state.pending = None
                    st.success("Demo action recorded. No money moved.")
                    st.rerun()

            if col2.button("Cancel plan", use_container_width=True):
                st.session_state.pending = None
                st.rerun()


# ---------- Transactions ----------
elif page == "Transactions":
    st.subheader("📋 Transaction activity")
    st.caption("Local demo records, not PayPal transaction history.")

    transactions = st.session_state.transactions

    if transactions:
        st.dataframe(
            [
                {
                    "Reference": t["id"],
                    "Recipient": t["recipient"],
                    "Purpose": t["purpose"],
                    "Amount": f"${t['amount']:,.2f}",
                    "Status": t["status"],
                    "Date": t["created"],
                }
                for t in transactions
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download activity JSON",
            data=json.dumps(transactions, indent=2),
            file_name="paypilot_activity.json",
            mime="application/json",
        )
    else:
        st.info("No transactions yet.")


# ---------- About ----------
elif page == "About":
    st.subheader("About PayPilot")
    st.write(
        "PayPilot explores how AI can turn payment requests into "
        "structured, reviewable workflows."
    )
    st.markdown("""
    **Current features**
    - AI-assisted request parsing
    - Local parser fallback
    - Payment validation
    - Explicit user approval
    - Session-based transaction history

    **Not implemented yet**
    - PayPal API integration
    - Real payment execution
    - Persistent transaction storage
    - Verified recipient identities
    """)
    st.info(
        "This is a hackathon prototype. All payment actions are simulated."
    )


st.divider()
st.caption(
    "PayPilot MVP · AI-assisted planning · "
    "All payments simulated · No money moves"
)