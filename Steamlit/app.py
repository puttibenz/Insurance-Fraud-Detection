import streamlit as st
import pandas as pd
import joblib

# 1. โหลดโมเดล
@st.cache_resource
def load_model():
    return joblib.load('C:\\Users\\Samak\\OneDrive\\เดสก์ท็อป\\Fraud Detection\\Steamlit\\fraud_lite_model.pkl')

package = load_model()
model = package['model']
threshold = package['threshold']
feature_cols = package['features']

# 2. หน้าตา Web App
st.set_page_config(page_title="Fraud Detection App", page_icon="🚨")
st.title("🚨 Auto Insurance Fraud Detection")
st.write("ระบบประเมินความเสี่ยงทุจริตการเคลมประกัน (Lite Model)")
st.divider()

st.subheader("📝 กรอกข้อมูลเพื่อประเมินความเสี่ยง")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("อายุผู้เอาประกัน (Age):", min_value=16, max_value=100, value=35)
    deductible = st.number_input("ค่าเสียหายส่วนแรก (Deductible):", min_value=0, value=400, step=100)
    year = st.selectbox("ปีที่เกิดเหตุ (Year):", [1994, 1995, 1996])
    month = st.selectbox("เดือนที่เกิดเหตุ (Month):", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    month_claimed = st.selectbox("เดือนที่แจ้งเคลม (Month Claimed):", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

with col2:
    fault = st.selectbox("ฝ่ายผิด (Fault):", ["Policy Holder", "Third Party"])
    base_policy = st.selectbox("แผนประกัน (Base Policy):", ["All Perils", "Collision", "Liability"])
    policy_type = st.selectbox("ประเภทกรมธรรม์ (Policy Type):", ["Sedan - Collision", "Other"])
    address_change = st.selectbox("การย้ายที่อยู่ (Address Change):", ['no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'])
    age_of_vehicle = st.selectbox("อายุรถยนต์ (Age of Vehicle):", ['new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'])

st.divider()

# 3. จัดการข้อมูลเมื่อกดปุ่มทำนาย
if st.button("🔍 ประเมินความเสี่ยง", use_container_width=True):
    
    # แปลงข้อความเป็นตัวเลข (ให้ตรงกับที่เทรนมา)
    month_dict = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    addr_dict = {'no change':0, 'under 6 months':1, '1 year':2, '2 to 3 years':3, '4 to 8 years':4}
    veh_age_dict = {'new':0, '2 years':1, '3 years':2, '4 years':3, '5 years':4, '6 years':5, '7 years':6, 'more than 7':7}
    
    # สร้าง DataFrame ขาเข้า
    input_data = pd.DataFrame(columns=feature_cols)
    input_data.loc[0] = 0 # ตั้งค่าเริ่มต้นเป็น 0
    
    # หยอดค่าลงไปใน 10 คอลัมน์
    input_data['Age'] = age
    input_data['Deductible'] = deductible
    input_data['Year'] = year
    input_data['Month'] = month_dict[month]
    input_data['MonthClaimed'] = month_dict[month_claimed]
    input_data['AddressChange_Claim'] = addr_dict[address_change]
    input_data['AgeOfVehicle'] = veh_age_dict[age_of_vehicle]
    
    # จัดการ One-hot features แบบง่ายๆ
    if fault == "Third Party": input_data['Fault_Third Party'] = 1
    if base_policy == "Liability": input_data['BasePolicy_Liability'] = 1
    if policy_type == "Sedan - Collision": input_data['PolicyType_Sedan - Collision'] = 1
    
    # ทำนายผล
    prob = model.predict_proba(input_data)[0][1]
    
    # แสดงผล
    if prob >= threshold:
        st.error(f"🚨 **ความเสี่ยงสูง!** มีโอกาสเป็นเคสทุจริต (ความน่าจะเป็น: {prob*100:.1f}%)")
        st.info(f"💡 เกณฑ์การแจ้งเตือน (Threshold) อยู่ที่ {threshold*100:.1f}%")
    else:
        st.success(f"✅ **ความเสี่ยงต่ำ** เคลมปกติ (ความน่าจะเป็น: {prob*100:.1f}%)")
        st.info(f"💡 เกณฑ์การแจ้งเตือน (Threshold) อยู่ที่ {threshold*100:.1f}%")