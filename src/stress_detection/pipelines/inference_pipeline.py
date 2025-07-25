import streamlit as st
import requests


class Infer_Orchestrator:
    def __init__(self, configs: dict):
        self.api_url = "http://127.0.0.1:8080/infer/"
        self.payload = {}

    def _render_form(self):
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
        st.title("Stress Level Prediction")
        st.success("Please read the README before using this app.", icon="ℹ️")
        st.sidebar.image('/workspace/image.png', use_container_width=True)
        with st.sidebar.expander("README"):
            st.markdown("Unlock the power of machine learning for mental well-being with our Stress Level Predictor! 🚦🤖 Effortlessly go from raw data to actionable insights as this project streamlines the entire MLOps cycle—automated processing, seamless deployment, and real-time monitoring—helping you predict stress like a pro. 🌟💡 Whether you're scaling in the cloud or experimenting locally, this repo makes stress detection reliable, efficient, and production-ready.")
        inference_tab, scripts_tab = st.tabs(["Inference", "Scripts"])

        with inference_tab:
            submit = self._render_form()
            if submit:
                with st.spinner("Sending data to model..."):
                    result, error = self._call_api()
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Predicted Stress Level: {result['stress_level']}")
        with scripts_tab:
            options = st.selectbox("Select Script", ["Feature Engineer", "Trainer"])
            if options == "Feature Engineer":
                if st.button("Trigger Feature Engineering Pipeline"):
                    from stress_detection.pipelines.feature_pipeline import fe_orchestrator
                    from stress_detection.scripts.utils import load_config
                    instance = fe_orchestrator(feature_configs=load_config("feature"))
                    st.warning("Started...")
                    instance.execute()
                    st.link_button(label="Redis Stack Browser", url="http://localhost:8001/redis-stack/browser")
                    st.success("Done...")
            elif options == "Trainer":
                if st.button("Trigger Training Pipeline"):
                    from stress_detection.pipelines.training_pipeline import train_orchestrator
                    from stress_detection.scripts.utils import load_config
                    instance = train_orchestrator(training_configs=load_config("training"))
                    try:
                        st.warning("Started...")
                        instance.execute()
                        st.link_button(label="Dagshub Mlflow Server", url="https://dagshub.com/Hg03/stress_detection.mlflow")
                        st.success("Done...")
                    except Exception as e:
                        st.error(f"Probably we don't have any model present. Please run the feature engineering pipeline first. {e}")

            

            
if __name__ == "__main__":
    inst = Infer_Orchestrator({})
    inst.execute()
