#include <SFML/Graphics.hpp>
#include <vector>
#include <string>
#include <sstream>

using namespace std;
using namespace sf;

// Function to map execution time to bar height
float mapToBarHeight(float time, float yMax, float windowHeight) {
    return (time / yMax) * (windowHeight - 100); // Adjust height for margin
}

void BarChart(vector<double> executionTimes) {
    const int windowWidth = 800;
    const int windowHeight = 600;
    const float barWidth = 60.0f; // Width of each bar
    const float barSpacing = 80.0f; // Space between bars

    sf::RenderWindow window(sf::VideoMode(windowWidth, windowHeight), "SFML Bar Chart with Execution Time");

    // Updated example execution times and labels to include 4 bars
    std::vector<std::string> labels = { " OMP", "CUDA", "MPI", "Single" };

    float yMax = 250.0f; // Maximum time in milliseconds for scaling (adjusted for new data)

    // Create the bar chart
    sf::RectangleShape bar(sf::Vector2f(barWidth, 0)); // Default bar size
    bar.setFillColor(sf::Color::Green);

    // Create text for labels
    sf::Font font;
    if (!font.loadFromFile("C:/Users/wongc/source/repos/DSCP/x64/Debug/arial.ttf")) {
        return; // Exit if font loading fails
    }

    sf::Text xLabel, timeLabel;
    xLabel.setFont(font);
    timeLabel.setFont(font);

    xLabel.setCharacterSize(14);
    timeLabel.setCharacterSize(14);

    xLabel.setFillColor(sf::Color::Black);
    timeLabel.setFillColor(sf::Color::Red); // Color for execution time labels

    // Set X-axis labels and execution time labels
    std::vector<sf::Text> xLabels;
    std::vector<sf::Text> timeLabels;
    for (size_t i = 0; i < labels.size(); ++i) {
        // X-axis label
        sf::Text labelText;
        labelText.setFont(font);
        labelText.setCharacterSize(14);
        labelText.setFillColor(sf::Color::Black);
        labelText.setString(labels[i]);
        labelText.setPosition(i * barSpacing + 40, windowHeight - 40); // Adjust x position for spacing

        xLabels.push_back(labelText);

        // Execution time label
        sf::Text timeText;
        timeText.setFont(font);
        timeText.setCharacterSize(14);
        timeText.setFillColor(sf::Color::Red);
        timeText.setString(std::to_string(static_cast<int>(executionTimes[i])) + " ms");
        timeText.setPosition(i * barSpacing + 40 + barWidth / 4, windowHeight - 50 - mapToBarHeight(executionTimes[i], yMax, windowHeight) - 20); // Above the bar

        timeLabels.push_back(timeText);
    }

    // Set up the bars
    std::vector<sf::RectangleShape> bars;
    for (size_t i = 0; i < executionTimes.size(); ++i) {
        bar.setSize(sf::Vector2f(barWidth, mapToBarHeight(executionTimes[i], yMax, windowHeight)));
        bar.setPosition(i * barSpacing + 20, windowHeight - 50 - bar.getSize().y); // Horizontal spacing and vertical positioning
        bars.push_back(bar);
    }

    // Main loop
    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        window.clear(sf::Color::White);

        // Draw bars
        for (const auto& b : bars) {
            window.draw(b);
        }

        // Draw X-axis labels
        for (const auto& l : xLabels) {
            window.draw(l);
        }

        // Draw execution time labels
        for (const auto& t : timeLabels) {
            window.draw(t);
        }

        window.display();
    }
   
}
