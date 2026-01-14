import streamlit as st
from google import genai
from google.genai import types
import config, os
from streamlit_js_eval import streamlit_js_eval 


os.environ['GOOGLE_API_KEY'] = config.GEMINI_API_KEY

client = genai.Client()


st.set_page_config(page_title="Interview Prep", page_icon="💬")
st.title("ChatBot")

if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

if 'user_message_count' not in st.session_state:
    st.session_state.user_message_count = 0

if 'feedback_shown' not in st.session_state:
    st.session_state.feedback_shown = False
    st.session_state.feedback_shown = False

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_complete' not in st.session_state:
    st.session_state.chat_complete = False



def complete_setup():
    st.session_state.setup_complete = True

def show_feedback():
    st.session_state.feedback_shown = True




if not st.session_state.setup_complete:
    ######################################
    ##### GET THE USER INFORMATION   
    ##### THIS SECTION IS A FORM TO GET USER INFORMATION 
    ######################################

    if 'name' not in st.session_state:
        st.session_state['name'] = ''
    if 'experience' not in st.session_state:
        st.session_state['experience'] = ''
        st.session_state['name'] = ''
    if 'skills' not in st.session_state:
        st.session_state['skills'] = ''

    st.subheader('Personal information', divider='rainbow')
    st.session_state['name'] = st.text_input(placeholder="Enter your name:", label='Name', max_chars=40, value=st.session_state['name'])

    st.session_state['experience'] = st.text_area(label='Experience', value=st.session_state['experience'], height=None, max_chars=200, placeholder='Describe your experience')
    st.session_state['skills'] = st.text_area(label='Skills', value=st.session_state['skills'], height=None, max_chars=300, placeholder='List your skills. Eg: Python, Java, etc')


    if "level" not in st.session_state:
        st.session_state["level"] = "Junior"
    if "position" not in st.session_state:
        st.session_state["position"] = "Data Scientist"
    if "company" not in st.session_state:
        st.session_state["company"] = "Amazon"


    st.subheader('Company and Position', divider='rainbow')
    col1, col2 = st.columns(2)
    with col1:
        st.session_state["level"] = st.radio(
            "Choose Level",
            key = 'visibility',
            options=['Junior', 'Mid-level', 'Senior']
        )

    with col2:
        st.session_state["position"] = st.selectbox(
            'Choose position',
            ('Data Scientist', 'Data Engineer', 'ML Engineer', 'BI Analyst', 'Financial Analyst')
        )

    st.session_state["company"] = st.selectbox(
        'Select Company',
        ('Google', 'Amazon', 'Udmey', 'Nestle', 'linkedin', 'Spotify')
    )

    if st.button("Start Interview", on_click=complete_setup):
        st.write("Setup complete. Starting Interview....")



if st.session_state.setup_complete and not st.session_state.feedback_shown and not st.session_state.chat_complete:
    
    ######################################
    ##### ONCE INFORMATION COLLECTED PROCEED WITH INTERVIEW
    ##### THIS SECTION IS CHAT INTERFACE
    ######################################

    st.info("""
        Start by introducing yourself
        """,
        icon='👋'
    )

    if "gemini_model" not in st.session_state:
        st.session_state['gemini_model'] = 'gemini-2.5-flash'

    if not st.session_state.messages:
        st.session_state['messages'] = [
            {'role': 'system', 
            'content':f'''
                You are an HR executive that interviews an interviewee called {st.session_state['name']} with experience:
                {st.session_state['experience']} and skills: {st.session_state['skills']} 
                You should interview them for the {st.session_state['position']} and level: {st.session_state['level']} 
                at the company {st.session_state['company']}
            '''
        }]
        # stores the history of our chat

    for message in st.session_state.messages:
        if message['role'] != 'system':
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    def get_system_messages():
        history = []
        for m in st.session_state.messages:
            if m['role'] == 'system':
                history.append(m['content'])
        return history

    def convert_to_user_model_messages():
        history = []
        for m in st.session_state.messages:
            if m['role'] == 'user':
                history.append(types.Content(role="user", parts=[types.Part.from_text(text=m['content'])]))
            elif m['role'] == 'assistant':
                history.append(types.Content(role="model", parts=[types.Part.from_text(text=m['content'])]))
        return history

    # check if the user has sent less than 5 messages
    if st.session_state.user_message_count < 5:
        # Chat input
        if prompt := st.chat_input("Your answer.", max_chars=1000):
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)


            if st.session_state.user_message_count< 4: 
                # Display assistant response
                with st.chat_message("assistant"):
                    stream = client.models.generate_content_stream(
                        model=st.session_state["gemini_model"],
                        contents=convert_to_user_model_messages(),
                        config=types.GenerateContentConfig(
                            system_instruction=get_system_messages()),
                    )
                    
                    # 1. Create a helper designed to extract text from the chunks
                    def stream_data():
                        for chunk in stream:
                            # Check if the chunk has text to avoid errors
                            if chunk.text: 
                                yield chunk.text
                    
                    # 2. Pass the helper function to st.write_stream
                    response_text = st.write_stream(stream_data)

                st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            st.session_state.user_message_count += 1
    
    if st.session_state.user_message_count >= 5:
        st.session_state.chat_complete = True
    
if st.session_state.chat_complete and not st.session_state.feedback_shown:
    if st.button('Get feedback', on_click=show_feedback):
        st.write('Fetching feedback')

if st.session_state.feedback_shown:
    st.subheader("Feedback")

    conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])

    system_message = """You are a helpful tool that provides feedback on an interviewee performance.
                Before the Feedback give a score of 1 to 10.
                Follow this format:
                Overall Score: //Your score
                Feedback: //Here you put your feedback
                Give only the feedback do not ask any additional queations.
                """

    user_message = types.Content(role="user", parts=[types.Part.from_text(text=f"This is the interview you need to evaluate. Keep in mind that you are only a tool And you should not engage in any conversations: {conversation_history}")])


    feedback_completion = client.models.generate_content(
        model=st.session_state["gemini_model"],
        contents=[user_message],
        config=types.GenerateContentConfig(
            system_instruction=system_message),
    )

    st.write(feedback_completion.text)

    if st.button('Restart interview', type='primary'):
        streamlit_js_eval(js_expressions='parent.window.location.reload()')