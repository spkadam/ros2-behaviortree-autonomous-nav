// Combined BehaviorTree + ROS2 Node Implementation for TurtleBot4
#include <memory>
#include <string>
#include <chrono>
#include <cmath>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"

#include "behaviortree_cpp_v3/action_node.h"
#include "behaviortree_cpp_v3/bt_factory.h"
#include "behaviortree_cpp_v3/loggers/bt_zmq_publisher.h"
#include "behaviortree_cpp_v3/loggers/bt_cout_logger.h"

using namespace std::chrono_literals;

// ============================================================================
// SHARED UTILITIES
// ============================================================================

static double computeMinForwardDistance(
    const sensor_msgs::msg::LaserScan& msg,
    double forward_angle = -(M_PI / 2.0),
    double angle_window = 20.0 * M_PI / 180.0)
{
    double min_dist = 999.0;
    int num_ranges = msg.ranges.size();
    for (int i = 0; i < num_ranges; ++i) {
        double angle = msg.angle_min + i * msg.angle_increment;
        if (std::abs(angle - forward_angle) <= angle_window) {
            double range = msg.ranges[i];
            if (std::isfinite(range) && range > msg.range_min && range < msg.range_max) {
                if (range < min_dist) {
                    min_dist = range;
                }
            }
        }
    }
    return min_dist;
}

// ============================================================================
// NODE DECLARATIONS
// ============================================================================

// Rotating - Rotates robot reading angle from blackboard
class Rotating : public BT::StatefulActionNode
{
public:
    Rotating(const std::string& name, const BT::NodeConfiguration& config,
             rclcpp::Node::SharedPtr node)
        : BT::StatefulActionNode(name, config), node_(node), angular_speed_(0.5)
    {
        cmd_vel_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    }

    static BT::PortsList providedPorts()
    {
        return { BT::InputPort<int>("input", "Rotation angle from blackboard interface") };
    }

    BT::NodeStatus onStart() override
    {
        int angle_degrees = 360;
        if (!getInput("input", angle_degrees)) {
            RCLCPP_WARN(node_->get_logger(), "No input angle, using default 360 degrees");
        }
        
        target_angle_ = angle_degrees * M_PI / 180.0;
        start_time_ = node_->now();
        
        RCLCPP_INFO(node_->get_logger(), "Rotating: target angle %.2f degrees", 
                    static_cast<double>(angle_degrees));
        
        return BT::NodeStatus::RUNNING;
    }

    BT::NodeStatus onRunning() override
    {
        auto elapsed = (node_->now() - start_time_).seconds();
        double angle_covered = angular_speed_ * elapsed;
        
        if (angle_covered >= std::abs(target_angle_)) {
            auto twist = geometry_msgs::msg::Twist();
            cmd_vel_pub_->publish(twist);
            RCLCPP_INFO(node_->get_logger(), "Rotation complete");
            return BT::NodeStatus::SUCCESS;
        }
        
        auto twist = geometry_msgs::msg::Twist();
        twist.angular.z = (target_angle_ > 0) ? angular_speed_ : -angular_speed_;
        cmd_vel_pub_->publish(twist);
        
        return BT::NodeStatus::RUNNING;
    }

    void onHalted() override
    {
        auto twist = geometry_msgs::msg::Twist();
        cmd_vel_pub_->publish(twist);
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Time start_time_;
    double target_angle_;
    double angular_speed_;
};

// ReadingLaserScanner - Reads and processes laser scan data
class ReadingLaserScanner : public BT::SyncActionNode
{
public:
    ReadingLaserScanner(const std::string& name, const BT::NodeConfiguration& config,
                        rclcpp::Node::SharedPtr node)
        : BT::SyncActionNode(name, config), node_(node)
    {
        scan_sub_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            [this](const sensor_msgs::msg::LaserScan::SharedPtr msg) {
                min_distance_.store(computeMinForwardDistance(*msg));
                scan_received_.store(true);
            });
    }

    static BT::PortsList providedPorts()
    {
        return { BT::OutputPort<double>("min_distance", "Minimum distance detected") };
    }

    BT::NodeStatus tick() override
    {
        if (!scan_received_.load()) {
            RCLCPP_WARN(node_->get_logger(), "Waiting for laser scan data...");
            return BT::NodeStatus::FAILURE;
        }
        
        double dist = min_distance_.load();
        RCLCPP_INFO(node_->get_logger(), "Laser scanner: minimum distance = %.2f m", dist);
        setOutput("min_distance", dist);
        
        return BT::NodeStatus::SUCCESS;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
    std::atomic<double> min_distance_{999.0};
    std::atomic<bool> scan_received_{false};
};

// MoveRobot - Moves robot forward for one tick (SyncActionNode)
class MoveRobot : public BT::SyncActionNode
{
public:
    MoveRobot(const std::string& name, const BT::NodeConfiguration& config,
              rclcpp::Node::SharedPtr node)
        : BT::SyncActionNode(name, config), node_(node), linear_speed_(0.2)
    {
        cmd_vel_pub_ = node_->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    }

    static BT::PortsList providedPorts()
    {
        return { BT::InputPort<double>("speed", 0.2, "Linear speed in m/s") };
    }

    BT::NodeStatus tick() override
    {
        getInput("speed", linear_speed_);
        auto twist = geometry_msgs::msg::Twist();
        twist.linear.x = linear_speed_;
        cmd_vel_pub_->publish(twist);
        return BT::NodeStatus::SUCCESS;
    }

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    double linear_speed_;
};

// IsObstacleFar - Condition node: succeeds if min_distance > 1.0
class IsObstacleFar : public BT::ConditionNode
{
public:
    IsObstacleFar(const std::string& name, const BT::NodeConfiguration& config)
        : BT::ConditionNode(name, config) {}

    static BT::PortsList providedPorts()
    {
        return { BT::InputPort<double>("min_distance", "Minimum distance to obstacle") };
    }

    BT::NodeStatus tick() override
    {
        double min_distance = 0.0;
        if (!getInput("min_distance", min_distance)) {
            std::cout << "[IsObstacleFar] No min_distance found on blackboard!" << std::endl;
            return BT::NodeStatus::FAILURE;
        }
        std::cout << "[IsObstacleFar] min_distance from blackboard: " << min_distance << std::endl;
        if (min_distance > 1.0) {
            std::cout << "[IsObstacleFar] SUCCESS: Obstacle is farther than 1m." << std::endl;
            return BT::NodeStatus::SUCCESS;
        }
        std::cout << "[IsObstacleFar] FAILURE: Obstacle is within 1m." << std::endl;
        return BT::NodeStatus::FAILURE;
    }
};


// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char** argv)
{
    // Initialize ROS2
    rclcpp::init(argc, argv);
    
    // Create a ROS2 node
    auto node = std::make_shared<rclcpp::Node>("turtlebot4_bt_node");
    
    RCLCPP_INFO(node->get_logger(), "Starting TurtleBot4 Behavior Tree Node");
    
    // Create BehaviorTree factory
    BT::BehaviorTreeFactory factory;
    
    // Register all custom nodes from the tree diagram
    BT::NodeBuilder rotating_builder = [node](const std::string& name, const BT::NodeConfiguration& config) {
        return std::make_unique<Rotating>(name, config, node);
    };
    factory.registerBuilder<Rotating>("Rotating", rotating_builder);

    BT::NodeBuilder scanner_builder = [node](const std::string& name, const BT::NodeConfiguration& config) {
        return std::make_unique<ReadingLaserScanner>(name, config, node);
    };
    factory.registerBuilder<ReadingLaserScanner>("ReadingLaserScanner", scanner_builder);

    BT::NodeBuilder move_builder = [node](const std::string& name, const BT::NodeConfiguration& config) {
        return std::make_unique<MoveRobot>(name, config, node);
    };
    factory.registerBuilder<MoveRobot>("MoveRobot", move_builder);

    // Register IsObstacleFar condition node
    factory.registerBuilder<IsObstacleFar>("IsObstacleFar", [](const std::string& name, const BT::NodeConfiguration& config) {
        return std::make_unique<IsObstacleFar>(name, config);
    });
    
    // Get parameters
    node->declare_parameter("bt_xml", "");
    node->declare_parameter("tree_name", "MainTree");
    
    std::string bt_xml_file;
    node->get_parameter("bt_xml", bt_xml_file);
    
    if (bt_xml_file.empty()) {
        // Use installed path
        bt_xml_file = "/home/administrator/ros2_ws/install/ros2_bt_turtlebot4/share/ros2_bt_turtlebot4/config/turtlebot4_bt.xml";
        RCLCPP_WARN(node->get_logger(), 
                    "No BT XML file specified, using default: %s", bt_xml_file.c_str());
    }
    
    // Create the behavior tree from XML
    BT::Tree tree;
    try {
        tree = factory.createTreeFromFile(bt_xml_file);
        RCLCPP_INFO(node->get_logger(), "Behavior Tree loaded successfully");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Failed to load Behavior Tree: %s", e.what());
        return 1;
    }
    
    // Enable Groot2 monitoring (optional - uncomment to use)
    BT::PublisherZMQ publisher_zmq(tree);
    
    // Add console logger for debugging
    BT::StdCoutLogger logger_cout(tree);
    
    // Create executor for ROS2 spinning
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    
    // Run the behavior tree
    RCLCPP_INFO(node->get_logger(), "Starting Behavior Tree execution");
    
    rclcpp::Rate rate(10); // 10 Hz

    while (rclcpp::ok()) {
        executor.spin_some();
        tree.tickRoot();
        rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
