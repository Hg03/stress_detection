import streamlit as st
import requests


class Infer_Orchestrator:
    def __init__(self, configs: dict):
        self.api_url = "http://127.0.0.1:8080/infer/"
        self.payload = {}

    def _render_form(self):
        st.title("Stress Level Prediction")
        with st.form("stress_form"):
            self.payload["employee_id"] = st.text_input("Employee ID", value="EMP_123")
            self.payload["avg_working_hours_per_day"] = st.slider("Average Working Hours Per Day", 0.0, 24.0, 8.0)
            self.payload["work_from"] = st.selectbox("Work From", ["Office", "Home", "Hybrid"])
            self.payload["work_pressure"] = st.slider("Work Pressure", 1, 5, 3)
            self.payload["manager_support"] = st.slider("Manager Support", 1, 5, 3)
            self.payload["sleeping_habit"] = st.slider("Sleeping Habit", 1, 5, 3)
            self.payload["exercise_habit"] = st.slider("Exercise Habit", 1, 5, 3)
            self.payload["job_satisfaction"] = st.slider("Job Satisfaction", 1, 5, 3)
            self.payload["work_life_balance"] = st.selectbox("Work-Life Balance", ["Yes", "No"])
            self.payload["social_person"] = st.slider("Social Person", 1, 5, 3)
            self.payload["lives_with_family"] = st.selectbox("Lives With Family", ["Yes", "No"])
            self.payload["working_state"] = st.selectbox("Working State", ["Hyderabad", "Karnataka", "Chennai", "Delhi", "Pune"])

            submit_button = st.form_submit_button("Predict Stress Level")

        return submit_button

    def _call_api(self):
        try:
            response = requests.post(self.api_url, data=self.payload)
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"API Error {response.status_code}: {response.text}"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {e}"

    def execute(self):
        submit = self._render_form()
        if submit:
            with st.spinner("Sending data to model..."):
                result, error = self._call_api()
                if error:
                    st.error(error)
                else:
                    st.success(f"Predicted Stress Level: {result['stress_level']}")
            
if __name__ == "__main__":
    inst = Infer_Orchestrator({})
    inst.execute()
