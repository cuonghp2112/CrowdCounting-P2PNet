## Running the Gradio Demo with Docker

To run the Gradio demo using Docker, follow these steps:

1. **Build the Docker Image:**
   ```bash
   docker build -t p2pnet-cpu .
   ```

2. **Run the Docker Container:**
   - Mount your local code directory to the container and expose a public port for the Gradio interface.
   ```bash
   docker run -p 7868:7860 -v $(pwd):/app -it --rm p2pnet-cpu bash
   ```

3. **Access the Demo:**
   - Open your web browser and navigate to `http://localhost:7860` to access the Gradio demo.

This setup allows you to run the demo in a containerized environment while still being able to modify the code locally.