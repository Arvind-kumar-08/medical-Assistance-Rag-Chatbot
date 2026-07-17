import streamlit as st

from utils.api import ask_question


def render_chat():
    st.subheader("💬 Chat with your assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your question")

    if not user_input:
        return

    # Save and display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get backend response
    try:
        with st.spinner("Thinking..."):
            response = ask_question(user_input)

        if response.status_code == 200:
            data = response.json()

            answer = data.get(
                "answer",
                "No answer received",
            )

            sources = data.get("sources", [])

            # Save assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            # Display assistant message
            with st.chat_message("assistant"):
                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            if isinstance(source, dict):
                                source_name = source.get(
                                    "source",
                                    "Unknown source",
                                )
                                page = source.get("page")

                                if page is not None:
                                    st.markdown(
                                        f"- {source_name}, page {page + 1}"
                                    )
                                else:
                                    st.markdown(f"- {source_name}")
                            else:
                                st.markdown(f"- {source}")

        else:
            try:
                error_data = response.json()
                error_message = error_data.get(
                    "error",
                    response.text,
                )
            except ValueError:
                error_message = response.text

            st.error(f"Error: {error_message}")

    except Exception as error:
        st.error(f"Unable to connect to the server: {error}")