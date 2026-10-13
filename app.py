
import json
import os
import uuid
from decimal import Decimal, InvalidOperation
from urllib.parse import urlencode

import requests
import streamlit as st
from groq import Groq


# ============================================================
# PAYPILOT — YOUR AI MONEY AGENT
# ============================================================

st.set_page_config(
    page_title="PayPilot | AI Money Agent",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

def get_secret(name, default=""):
    try:
        value = st.secrets.get(name, default)
        return value if value else os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

PAYPAL_CLIENT_ID = get_secret("PAYPAL_CLIENT_ID")
PAYPAL_CLIENT_SECRET = get_secret("PAYPAL_CLIENT_SECRET")
PAYPAL_MODE = get_secret("PAYPAL_MODE", "sandbox").lower()
APP_URL = get_secret("APP_URL", "http://localhost:8501").rstrip("/")

if PAYPAL_MODE not in ("sandbox", "live"):
    PAYPAL_MODE = "sandbox"

# Deliberately disable live payments in this prototype.
LIVE_PAYMENTS_ENABLED = False

PAYPAL_BASE = (
    "https://api-m.sandbox.paypal.com"
    if PAYPAL_MODE == "sandbox"
    else "https://api-m.paypal.com"
)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "navigation": "Overview",
    "budget": {
        "income": 3000.0,
        "bills": 1200.0,
        "savings": 300.0,
    },
    "shopping_lists": [],
    "active_list_id": None,
    "plan": None,
    "order": None,
    "activity": [
        {
            "id": "SAMPLE-1001",
            "recipient": "Jordan Lee",
            "amount": 28.50,
            "purpose": "Team lunch",
            "status": "Illustrative sample",
        },
        {
            "id": "SAMPLE-1002",
            "recipient": "Cloud Studio",
            "amount": 14.00,
            "purpose": "Design subscription",
            "status": "Illustrative sample",
        },
    ],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value.copy() if isinstance(value, dict) else (
            list(value) if isinstance(value, list) else value
        )


# ============================================================
# GROQ AI
# ============================================================

def get_groq_client():
    if not GROQ_API_KEY:
        raise ValueError(
            "Groq API key is missing. Add GROQ_API_KEY to "
            ".streamlit/secrets.toml."
        )
    return Groq(api_key=GROQ_API_KEY)


def make_payment_plan(text):
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract a proposed payment request. Return JSON with "
                    "recipient (string), amount (number or null), and "
                    "purpose (string). Never execute payments. Never invent "
                    "missing details. The amount must be positive. If the "
                    "request is ambiguous, explain that details are missing "
                    "rather than making up an amount or recipient."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    data = json.loads(response.choices[0].message.content)

    recipient = str(data.get("recipient") or "").strip()
    purpose = str(data.get("purpose") or "General payment").strip()

    if not recipient or len(recipient) > 80:
        raise ValueError("Please specify a valid payment recipient.")

    try:
        amount = Decimal(str(data.get("amount")))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("Please include a clear payment amount.")

    if not amount.is_finite() or not Decimal("0") < amount <= Decimal("10000"):
        raise ValueError("Enter an amount between $0.01 and $10,000.")

    return {
        "recipient": recipient,
        "amount": float(amount.quantize(Decimal("0.01"))),
        "purpose": purpose[:120],
    }


def generate_shopping_list(request_text, budget_limit):
    client = get_groq_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PayPilot's practical shopping assistant. "
                    "Return JSON with an items array. Each item must contain "
                    "name (string), category (string), quantity (string), "
                    "and estimated_price (number in USD). Generate up to "
                    "25 items. Prices are estimates, not live retailer "
                    "prices. Aim to stay within budget when practical. "
                    "Never claim to have checked live prices."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "request": request_text,
                    "budget_usd": round(float(budget_limit), 2),
                }),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    data = json.loads(response.choices[0].message.content)
    raw_items = data.get("items", [])

    if not isinstance(raw_items, list):
        raise ValueError("The AI returned an invalid shopping list.")

    items = []

    for item in raw_items[:25]:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name") or "").strip()[:100]
        category = str(item.get("category") or "Other").strip()[:50]
        quantity = str(item.get("quantity") or "1").strip()[:50]

        try:
            price = Decimal(str(item.get("estimated_price")))
        except (InvalidOperation, TypeError, ValueError):
            continue

        if (
            not name
            or not price.is_finite()
            or price < 0
            or price > Decimal("100000")
        ):
            continue

        items.append({
            "id": uuid.uuid4().hex[:10],
            "name": name,
            "category": category,
            "quantity": quantity,
            "estimated_price": float(price.quantize(Decimal("0.01"))),
            "purchased": False,
        })

    if not items:
        raise ValueError(
            "No valid items were generated. Try a more specific request."
        )

    return items


def generate_budget_insights(budget, lists):
    client = get_groq_client()

    totals = calculate_budget(budget)

    shopping_summary = []
    for entry in lists:
        shopping_summary.append({
            "request": entry["request"],
            "budget": entry["budget"],
            "estimated_total": round(
                sum(i["estimated_price"] for i in entry["items"]), 2
            ),
            "checked_off_total": round(
                sum(
                    i["estimated_price"]
                    for i in entry["items"]
                    if i["purchased"]
                ),
                2,
            ),
        })

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are PayPilot, a practical budgeting assistant. "
                    "Give concise, helpful observations and suggestions. "
                    "Clearly distinguish user-entered budget values from "
                    "estimated shopping costs. Do not claim to have access "
                    "to bank accounts, PayPal balances, or actual spending. "
                    "Do not present estimates as confirmed transactions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "budget": totals,
                    "shopping_lists": shopping_summary,
                }),
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


# ============================================================
# BUDGET CALCULATIONS
# ============================================================

def calculate_budget(budget):
    income = Decimal(str(budget["income"]))
    bills = Decimal(str(budget["bills"]))
    savings = Decimal(str(budget["savings"]))

    return {
        "income": float(income),
        "bills": float(bills),
        "savings": float(savings),
        "available": float(income - bills - savings),
    }


# ============================================================
# PAYPAL CHECKOUT API
# ============================================================

def paypal_access_token():
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise ValueError(
            "Add PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET "
            "to .streamlit/secrets.toml."
        )

    response = requests.post(
        f"{PAYPAL_BASE}/v1/oauth2/token",
        auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET),
        headers={"Accept": "application/json"},
        data={"grant_type": "client_credentials"},
        timeout=25,
    )

    if not response.ok:
        raise RuntimeError(
            f"PayPal authentication failed (HTTP {response.status_code}). "
            "Check your credentials and environment."
        )

    return response.json()["access_token"]


def create_paypal_order(amount, description):
    token = paypal_access_token()

    if PAYPAL_MODE == "live" and not LIVE_PAYMENTS_ENABLED:
        raise ValueError("Live payments are disabled in this prototype.")

    return_url = f"{APP_URL}/?{urlencode({'paypal_return': '1'})}"
    cancel_url = f"{APP_URL}/?{urlencode({'paypal_cancel': '1'})}"

    response = requests.post(
        f"{PAYPAL_BASE}/v2/checkout/orders",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "intent": "CAPTURE",
            "purchase_units": [
                {
                    "description": description[:127],
                    "amount": {
                        "currency_code": "USD",
                        "value": f"{amount:.2f}",
                    },
                }
            ],
            "application_context": {
                "brand_name": "PayPilot Demo",
                "user_action": "PAY_NOW",
                "return_url": return_url,
                "cancel_url": cancel_url,
            },
        },
        timeout=25,
    )

    if not response.ok:
        raise RuntimeError(
            f"PayPal order creation failed (HTTP {response.status_code}). "
            "Check your Sandbox app permissions and credentials."
        )

    order = response.json()
    approval_url = next(
        (
            link["href"]
            for link in order.get("links", [])
            if link.get("rel") == "approve"
        ),
        None,
    )

    if not approval_url:
        raise RuntimeError("PayPal did not return an approval URL.")

    return {
        "id": order["id"],
        "amount": float(amount),
        "description": description,
        "approval_url": approval_url,
    }


def capture_paypal_order(order_id):
    token = paypal_access_token()

    response = requests.post(
        f"{PAYPAL_BASE}/v2/checkout/orders/{order_id}/capture",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={},
        timeout=25,
    )

    if not response.ok:
        try:
            message = response.json().get("message", "Capture failed")
        except ValueError:
            message = "Capture failed"
        raise RuntimeError(
            f"PayPal capture failed (HTTP {response.status_code}): {message}"
        )

    return response.json()


def verify_paypal_order(order_id):
    token = paypal_access_token()

    response = requests.get(
        f"{PAYPAL_BASE}/v2/checkout/orders/{order_id}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=25,
    )

    if not response.ok:
        raise RuntimeError("Could not verify the PayPal order status.")

    return response.json()


# ============================================================
# STYLING
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #edf2ff;
}
[data-testid="stSidebar"] {
    background: #11182b;
    border-right: 1px solid #25334f;
}
[data-testid="stMetric"] {
    background: #151e33;
    border: 1px solid #293652;
    border-radius: 14px;
    padding: 15px;
}
.hero {
    background: linear-gradient(120deg,#182c53,#172044 60%,#291b4b);
    border: 1px solid #33466e;
    border-radius: 20px;
    padding: 26px;
    margin-bottom: 22px;
}
.eyebrow {
    color: #91a8ff;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: 700;
}
.muted {
    color: #a9b6d0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PAYPAL REDIRECT HANDLING
# ============================================================

query_params = st.query_params

if query_params.get("paypal_cancel") == "1":
    st.warning("Checkout cancelled. No capture was requested.")
    st.query_params.clear()

if query_params.get("paypal_return") == "1":
    returned_id = query_params.get("token", "")
    current_order = st.session_state.order

    if current_order and returned_id == current_order["id"]:
        try:
            verified = verify_paypal_order(returned_id)

            if verified.get("status") == "APPROVED":
                st.session_state.returned_order_id = returned_id
                st.session_state.order_verified = True
            elif verified.get("status") == "COMPLETED":
                st.session_state.returned_order_id = returned_id
                st.session_state.order_verified = True
                st.session_state.order_already_completed = True
            else:
                st.session_state.order_verified = False
                st.warning(
                    f"PayPal reports order status: {verified.get('status')}."
                )
        except Exception as exc:
            st.session_state.order_verified = False
            st.error(f"Could not verify returned order: {exc}")
    else:
        st.session_state.order_verified = False
        st.error(
            "The returned order does not match the current browser session. "
            "Create a new sandbox checkout."
        )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

PAGES = [
    "Overview",
    "AI Assistant",
    "Budget & Lists",
    "PayPal Checkout",
    "AI Insights",
    "Activity",
]

with st.sidebar:
    st.markdown("# 💸 PayPilot")
    st.caption("YOUR AI MONEY AGENT")
    st.divider()

    st.radio(
        "WORKSPACE",
        PAGES,
        key="navigation",
        on_change=lambda: None,
    )

    st.divider()
    st.caption(f"Groq AI: {'Configured' if GROQ_API_KEY else 'Not configured'}")
    st.caption(f"PayPal: {PAYPAL_MODE.upper()}")
    st.caption("Personal wallet connection: Not implemented")
    st.caption("Budget data: Current session only")


page = st.session_state.navigation


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="eyebrow">AI · PAYMENTS · HUMAN CONTROL</div>
    <h1>PayPilot — Your AI Money Agent</h1>
    <p>Plan your money. Build better habits. Review every payment.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.subheader("Your financial command center")

    st.warning(
        "Prototype notice: sample activity is illustrative. "
        "This app is not connected to a personal PayPal wallet."
    )

    budget = calculate_budget(st.session_state.budget)
    lists = st.session_state.shopping_lists

    shopping_estimate = sum(
        sum(item["estimated_price"] for item in entry["items"])
        for entry in lists
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Available to plan", f"${budget['available']:,.2f}")
    c2.metric("Shopping lists", len(lists))
    c3.metric("Shopping estimates", f"${shopping_estimate:,.2f}")

    st.markdown("### What would you like to do?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🤖 AI Assistant")
        st.write("Turn a natural-language request into a payment plan.")
        if st.button("Create payment plan →", key="home_payment", type="primary"):
            st.session_state.navigation = "AI Assistant"
            st.rerun()

    with col2:
        st.markdown("#### 💰 Budget & Lists")
        st.write("Plan monthly spending and create shopping lists.")
        if st.button("Open Budget & Lists →", key="home_budget"):
            st.session_state.navigation = "Budget & Lists"
            st.rerun()

    with col3:
        st.markdown("#### 🔗 PayPal Checkout")
        st.write("Review and approve a sandbox checkout.")
        if st.button("Open PayPal Checkout →", key="home_checkout"):
            st.session_state.navigation = "PayPal Checkout"
            st.rerun()

    st.markdown("### Monthly allocation")

    if budget["available"] < 0:
        st.error(
            f"Your bills and savings goal exceed your income by "
            f"${abs(budget['available']):,.2f}."
        )
    elif budget["income"] > 0:
        allocated = budget["income"] - budget["available"]
        st.progress(
            min(1.0, max(0.0, allocated / budget["income"])),
            text="Income allocated to bills and savings",
        )

    st.caption(
        "Budget amounts are based on your inputs. Shopping totals use "
        "estimated prices and do not represent verified spending."
    )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "AI Assistant":
    st.subheader("🤖 AI Payment Planner")
    st.write(
        "Describe a payment. PayPilot extracts the details for you to review."
    )

    with st.form("payment_form"):
        request_text = st.text_area(
            "Describe your payment",
            placeholder="Prepare a $35 checkout for a design consultation",
        )
        submit_plan = st.form_submit_button(
            "Create payment plan",
            type="primary",
        )

    if submit_plan:
        try:
            st.session_state.plan = make_payment_plan(request_text)
            st.success("Payment plan prepared.")
        except Exception as exc:
            st.error(f"Could not prepare payment plan: {exc}")

    plan = st.session_state.plan

    if plan:
        st.markdown("### Review payment details")

        c1, c2 = st.columns(2)
        c1.metric("Amount", f"${plan['amount']:,.2f} USD")
        c2.write(f"**Recipient / description**\n\n{plan['recipient']}")

        st.write(f"**Purpose:** {plan['purpose']}")

        st.info(
            "Creating this plan does not move money. The PayPal integration "
            "below creates a merchant checkout order, not a personal "
            "person-to-person transfer."
        )

        if st.button(
            "Create PayPal sandbox checkout →",
            type="primary",
        ):
            try:
                order = create_paypal_order(
                    plan["amount"],
                    f"{plan['recipient']} — {plan['purpose']}",
                )
                st.session_state.order = order
                st.session_state.order_verified = False
                st.session_state.returned_order_id = None
                st.session_state.order_already_completed = False
                st.session_state.navigation = "PayPal Checkout"
                st.rerun()
            except Exception as exc:
                st.error(f"Could not create PayPal order: {exc}")


# ============================================================
# BUDGET & LISTS
# ============================================================

elif page == "Budget & Lists":
    st.subheader("💰 Smart Budget & Lists")
    st.caption(
        "Manage your monthly budget and generate shopping lists with Groq."
    )

    budget = st.session_state.budget

    with st.form("budget_form"):
        st.markdown("### Your monthly budget")
        c1, c2, c3 = st.columns(3)

        income = c1.number_input(
            "Monthly take-home income ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=float(budget["income"]),
            step=100.0,
        )

        bills = c2.number_input(
            "Monthly bills ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=float(budget["bills"]),
            step=50.0,
        )

        savings = c3.number_input(
            "Monthly savings goal ($)",
            min_value=0.0,
            max_value=10000000.0,
            value=float(budget["savings"]),
            step=50.0,
        )

        save_budget = st.form_submit_button(
            "Save monthly budget",
            type="primary",
        )

    if save_budget:
        st.session_state.budget = {
            "income": round(income, 2),
            "bills": round(bills, 2),
            "savings": round(savings, 2),
        }
        st.rerun()

    totals = calculate_budget(st.session_state.budget)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income", f"${totals['income']:,.2f}")
    c2.metric("Bills", f"${totals['bills']:,.2f}")
    c3.metric("Savings", f"${totals['savings']:,.2f}")
    c4.metric("Available", f"${totals['available']:,.2f}")

    if totals["available"] < 0:
        st.error(
            "Your bills and savings goal exceed your income. "
            "Review your budget before adding more spending."
        )
    elif totals["income"] > 0:
        allocated = totals["income"] - totals["available"]
        st.progress(
            min(1.0, max(0.0, allocated / totals["income"])),
            text="Income allocated",
        )

    st.divider()
    st.markdown("### Generate a shopping list")

    with st.form("shopping_form"):
        request = st.text_area(
            "What do you need?",
            placeholder=(
                "Groceries for one person for 7 days, including meals, "
                "fruit, snacks, and household essentials."
            ),
        )

        default_limit = max(0.0, min(100.0, totals["available"]))

        shopping_budget = st.number_input(
            "Shopping budget ($)",
            min_value=0.01,
            max_value=100000.0,
            value=max(0.01, float(default_limit)),
            step=5.0,
        )

        generate = st.form_submit_button(
            "✨ Generate shopping list",
            type="primary",
        )

    if generate:
        if not request.strip():
            st.warning("Describe what you need first.")
        else:
            try:
                with st.spinner("Groq is building your shopping list..."):
                    items = generate_shopping_list(
                        request.strip(),
                        shopping_budget,
                    )

                new_list = {
                    "id": uuid.uuid4().hex[:8],
                    "request": request.strip(),
                    "budget": round(shopping_budget, 2),
                    "items": items,
                }

                st.session_state.shopping_lists.insert(0, new_list)
                st.session_state.active_list_id = new_list["id"]
                st.success("Shopping list created.")
                st.rerun()

            except Exception as exc:
                st.error(f"Shopping list generation failed: {exc}")

    st.divider()
    st.markdown("### Your lists")

    lists = st.session_state.shopping_lists

    if not lists:
        st.info("You haven't created a shopping list yet.")
    else:
        options = {
            f"{entry['request'][:45]} — ${entry['budget']:.2f}": entry["id"]
            for entry in lists
        }

        labels = list(options.keys())
        default_label = next(
            (
                label for label, list_id in options.items()
                if list_id == st.session_state.active_list_id
            ),
            labels[0],
        )

        selected_label = st.selectbox(
            "Select list",
            labels,
            index=labels.index(default_label),
            key="shopping_list_selector",
        )

        selected_id = options[selected_label]

        current_list = next(
            entry for entry in lists if entry["id"] == selected_id
        )

        items = current_list["items"]

        estimated_total = round(
            sum(item["estimated_price"] for item in items), 2
        )
        checked_total = round(
            sum(
                item["estimated_price"]
                for item in items
                if item["purchased"]
            ),
            2,
        )

        remaining = current_list["budget"] - checked_total

        c1, c2, c3 = st.columns(3)
        c1.metric("Estimated list total", f"${estimated_total:.2f}")
        c2.metric("Checked-off total", f"${checked_total:.2f}")
        c3.metric("Budget remaining", f"${remaining:.2f}")

        if estimated_total > current_list["budget"]:
            st.warning(
                f"List exceeds its budget by "
                f"${estimated_total - current_list['budget']:.2f}."
            )
        else:
            st.success(
                f"List is ${current_list['budget'] - estimated_total:.2f} "
                "under budget."
            )

        if items:
            progress = sum(
                1 for item in items if item["purchased"]
            ) / len(items)

            st.progress(progress, text="Shopping progress")

            for item in items:
                checkbox_key = f"buy_{current_list['id']}_{item['id']}"

                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = item["purchased"]

                left, middle, right = st.columns([0.12, 0.62, 0.26])

                checked = left.checkbox(
                    "Bought",
                    key=checkbox_key,
                    label_visibility="collapsed",
                )

                item["purchased"] = checked

                middle.markdown(
                    f"**{item['name']}**  \n"
                    f"<span style='color:#a9b6d0;font-size:13px'>"
                    f"{item['category']} · {item['quantity']}</span>",
                    unsafe_allow_html=True,
                )

                right.markdown(f"**${item['estimated_price']:.2f}**")

        st.download_button(
            "Download shopping list JSON",
            data=json.dumps(current_list, indent=2),
            file_name=f"paypilot-list-{current_list['id']}.json",
            mime="application/json",
        )

        if st.button("Delete selected list"):
            st.session_state.shopping_lists = [
                entry for entry in lists if entry["id"] != selected_id
            ]
            st.session_state.active_list_id = (
                st.session_state.shopping_lists[0]["id"]
                if st.session_state.shopping_lists
                else None
            )
            st.rerun()

    st.caption(
        "Prices are AI estimates, not live retailer prices. Checking an "
        "item does not make a payment. Data is held in the current session."
    )


# ============================================================
# PAYPAL CHECKOUT
# ============================================================

elif page == "PayPal Checkout":
    st.subheader("🔗 PayPal Sandbox Checkout")

    st.warning(
        "This is merchant checkout, not personal wallet access or a "
        "person-to-person transfer. Sandbox transactions use test money."
    )

    order = st.session_state.order

    if not order:
        st.info("Create a payment plan first in AI Assistant.")
    else:
        st.write(f"**Order ID:** `{order['id']}`")
        st.write(f"**Description:** {order['description']}")
        st.metric("Order amount", f"${order['amount']:.2f} USD")

        st.link_button(
            "Open PayPal approval page",
            order["approval_url"],
        )

        returned_id = st.session_state.get("returned_order_id")
        verified = st.session_state.get("order_verified", False)

        if returned_id == order["id"] and verified:
            if st.session_state.get("order_already_completed"):
                st.success("PayPal reports this order is already completed.")
            else:
                st.success("The returned order was verified with PayPal.")

                consent = st.checkbox(
                    "I have reviewed this sandbox order and want to request capture."
                )

                if st.button(
                    "Confirm and capture sandbox payment",
                    type="primary",
                ):
                    if not consent:
                        st.error("Confirm the checkbox before capturing.")
                    else:
                        try:
                            result = capture_paypal_order(order["id"])
                            status = result.get("status", "UNKNOWN")

                            st.write(f"PayPal capture status: **{status}**")

                            if status == "COMPLETED":
                                already_recorded = any(
                                    entry["id"] == order["id"]
                                    for entry in st.session_state.activity
                                )

                                if not already_recorded:
                                    st.session_state.activity.insert(
                                        0,
                                        {
                                            "id": order["id"],
                                            "recipient": order["description"],
                                            "amount": order["amount"],
                                            "purpose": "PayPal sandbox checkout",
                                            "status": "Completed sandbox capture",
                                        },
                                    )

                                st.success(
                                    "PayPal confirmed the sandbox capture."
                                )
                            else:
                                st.warning(
                                    "The order is not marked completed. "
                                    "Check PayPal's returned status."
                                )

                        except Exception as exc:
                            st.error(f"Capture failed: {exc}")
        else:
            st.info(
                "Approve the order on PayPal, then return to this app. "
                "PayPilot verifies the returned order before capture."
            )


# ============================================================
# AI INSIGHTS
# ============================================================

elif page == "AI Insights":
    st.subheader("📊 AI Budget Insights")
    st.caption(
        "Insights use your entered budget and the shopping lists in this session."
    )

    totals = calculate_budget(st.session_state.budget)

    c1, c2, c3 = st.columns(3)
    c1.metric("Income", f"${totals['income']:,.2f}")
    c2.metric("Bills", f"${totals['bills']:,.2f}")
    c3.metric("Available", f"${totals['available']:,.2f}")

    if st.button("Generate AI insights", type="primary"):
        try:
            with st.spinner("Groq is reviewing your budget..."):
                insights = generate_budget_insights(
                    st.session_state.budget,
                    st.session_state.shopping_lists,
                )
            st.markdown(insights)
        except Exception as exc:
            st.error(f"Could not generate insights: {exc}")

    st.info(
        "Insights do not include personal PayPal account history. "
        "Shopping prices and suggestions are estimates."
    )


# ============================================================
# ACTIVITY
# ============================================================

elif page == "Activity":
    st.subheader("📋 Activity History")
    st.caption(
        "Local app records only. This is not a full PayPal account statement."
    )

    activity = st.session_state.activity

    if activity:
        st.dataframe(
            activity,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Export activity JSON",
            data=json.dumps(activity, indent=2),
            file_name="paypilot-activity.json",
            mime="application/json",
        )
    else:
        st.info("No activity records yet.")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "PayPilot MVP · Groq AI · PayPal REST API · Sandbox checkout · "
    "Session-based budgeting · No personal wallet connection"
)
