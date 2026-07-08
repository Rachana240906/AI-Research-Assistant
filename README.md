This repository contains the full source code and interface design for a modern AI-powered research assistant dashboard. 
The goal of this project is to take raw, unformatted text outputs from large language models and automatically transform them into a highly organized, three-column analytics workspace. 
The application is built around a sleek dark-mode glassmorphic aesthetic that focuses heavily on clean content scannability and technical document layout.
The front-end design splits the viewport into three distinct vertical panels to maximize reading efficiency. 
The left column serves as a document map that displays a sticky table of contents, allowing users to quickly jump between different sections of the synthesized report, and includes quick action buttons for exporting text. 
The middle column acts as the primary reading pane, featuring meta tags for tracking topics, an executive summary callout box for rapid reviews, the main compiled intelligence report, and a structured citation grid for verified source materials. 
The right column handles live telemetry statistics, tracking confidence scoring models and source depth to give real-time feedback on the generation process.
A major focus of this project was optimizing the user experience through precise typography and front-end optimization. The entire interface is styled uniformly using the space-efficient Roboto Condensed font family by Christian Robertson to ensure heavy research text remains highly readable. 
The user interface also includes a unique floating ribbon sticker component positioned directly over the search input terminal that loops a smooth, persistent rotational swinging animation using CSS keyframes. 
For data processing, the front-end features a custom JavaScript markdown parser that instantly compiles bold weights, structural headings, and list items on the fly. 
It also includes a dual-mode secure clipboard fallback function that allows the copy-text feature to work perfectly in non-secure local testing environments by quietly creating and executing text actions on temporary background elements.
The repository is organized into a clean decoupled structure with a clear separation of concerns. 
The backend directory contains the FastAPI routing infrastructure and the core LangChain generation logic services. 
The frontend directory holds the structural HTML document, the custom stylesheet outlining the layout grids, and the primary application script managing the API connections and user interactions. 
To run the project locally, the backend can be spun up using a simple pip installation and a uvicorn terminal command, while the frontend is best served using a standard Python local HTTP server script to prevent local browser security policies from blocking the clipboard and layout functions.
