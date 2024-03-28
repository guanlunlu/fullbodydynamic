#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>

typedef Eigen::Matrix<double, 6, 1> Vector6d;

Eigen::Matrix3d skewSymMat(const Eigen::Vector3d &vx)
{
    Eigen::Matrix3d result;
    result << 0, -vx(2), vx(1),
        vx(2), 0, -vx(0),
        -vx(1), vx(0), 0;
    return result;
}

class Twist
{
public:
    Eigen::Vector3d v;
    Eigen::Vector3d w;
    Eigen::Matrix<double, 6, 1> Mat; // Assuming Mat needs to be accessed publicly

    // Constructor
    Twist(const Eigen::Vector3d &v, const Eigen::Vector3d &w) : v(v), w(w)
    {
        Mat << w, v; // Equivalent to np.block, arranging w and v vertically
    }

    // Adjoint method
    Eigen::Matrix<double, 6, 6> adjoint() const
    {
        Eigen::Matrix<double, 6, 6> result;
        result.block<3, 3>(0, 0) = skewSymMat(w);
        result.block<3, 3>(0, 3).setZero();
        result.block<3, 3>(3, 0) = skewSymMat(v);
        result.block<3, 3>(3, 3) = skewSymMat(w);
        return result;
    }

    // Convert frame method
    Twist convertFrame(const Twist &T_base2target)
    {
        Eigen::Matrix<double, 6, 1> v_new = T_base2target.adjoint() * Mat;
        return Twist(v_new.segment<3>(3), v_new.segment<3>(0));
    }

    // Integrate method
    Twist integrate(const Eigen::Vector3d &dvdt, const Eigen::Vector3d &dwdt, double dt = 0.001)
    {
        return Twist(v + dvdt * dt, w + dwdt * dt);
    }
};

class HomoTransform
{
public:
    Eigen::Matrix3d R;   // Rotation matrix
    Eigen::Vector3d p;   // Translation vector
    Eigen::Matrix4d Mat; // Homogeneous transformation matrix

    // Constructor
    HomoTransform(const Eigen::Matrix3d &R, const Eigen::Vector3d &p) : R(R), p(p)
    {
        Mat.setZero();
        Mat.block<3, 3>(0, 0) = R;
        Mat.block<3, 1>(0, 3) = p;
        Mat(3, 3) = 1.0;
    }

    // Adjoint matrix of the transformation
    Eigen::Matrix<double, 6, 6> adjoint() const
    {
        Eigen::Matrix<double, 6, 6> result;
        result.block<3, 3>(0, 0) = R;
        result.block<3, 3>(0, 3).setZero();
        result.block<3, 3>(3, 0) = skewSymMat(p) * R;
        result.block<3, 3>(3, 3) = R;
        return result;
    }

    // Inverse of the transformation
    HomoTransform inverse() const
    {
        Eigen::Matrix3d t11 = R.transpose();
        Eigen::Vector3d t12 = -t11 * p;
        return HomoTransform(t11, t12);
    }

    // Method to integrate a twist over time dt
    HomoTransform integrate(const Twist &twist_b, double dt = 0.001)
    {
        double w_norm = twist_b.w.norm();
        Eigen::Matrix3d rotm;
        if (w_norm > 1e-7)
        {
            // Eigen doesn't directly support rotation vectors, so we convert to quaternion and back
            Eigen::Quaterniond q;
            q = Eigen::AngleAxisd(w_norm * dt, twist_b.w.normalized());
            rotm = q.toRotationMatrix();
        }
        else
        {
            rotm.setIdentity();
        }

        Eigen::Matrix3d R_n = rotm * R;
        Eigen::Vector3d p_n = twist_b.v * dt + p;
        return HomoTransform(R_n, p_n);
    }

    void updateMat(const Eigen::Matrix3d &R_, const Eigen::Vector3d &p_)
    {
        R = R_;
        p = p_;
        Mat.setZero();
        Mat.block<3, 3>(0, 0) = R;
        Mat.block<3, 1>(0, 3) = p;
        Mat(3, 3) = 1.0;
    }
};

class Wrench
{
public:
    Eigen::Vector3d m;               // Moment vector
    Eigen::Vector3d f;               // Force vector
    Eigen::Matrix<double, 6, 1> Mat; // Combined wrench vector

    // Constructor
    Wrench(const Eigen::Vector3d &moment, const Eigen::Vector3d &force) : m(moment), f(force)
    {
        Mat << m, f; // Combine moment and force into a single vector
    }

    // Convert wrench from current base frame to target frame
    Wrench convertFrame(const HomoTransform &T_base2target)
    {
        Eigen::Matrix<double, 6, 1> F_target = T_base2target.adjoint().transpose() * Mat;
        return Wrench(F_target.segment<3>(0), F_target.segment<3>(3));
    }
};

class FloatingBase
{
public:
    double m = 21;
    double g = -9.81;
    Eigen::Matrix3d Ib;             // Inertia matrix in body frame
    Eigen::Matrix<double, 6, 6> Mq; // Generalized inertia matrix
    HomoTransform T_sb;             // Transformation from space to body frame
    Twist V_b;                      // Body twist

    FloatingBase(const Eigen::Vector3d &init_p, const Eigen::Matrix3d &init_R, const Twist &init_vb) : T_sb(init_R, init_p), V_b(init_vb)
    {
        // Initialize physical properties
        double w = 577.5 * 0.001;
        double h = 144 * 0.001;
        double l = 329.5 * 0.001;
        double Ixx = m * (w * w + h * h) / 12.0;
        double Iyy = m * (l * l + h * h) / 12.0;
        double Izz = m * (l * l + w * w) / 12.0;
        Ib = Eigen::Matrix3d::Zero();
        Ib.diagonal() << Ixx, Iyy, Izz;

        // Construct generalized inertia matrix
        Mq.setZero();
        Mq.block<3, 3>(0, 0) = Ib;
        Mq.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * m;
    }

    Wrench gravityWrench(const HomoTransform &T_sb)
    {
        // Gravity force vector in space frame
        Eigen::Vector3d mg_s(0, 0, m * g);

        // Calculate the moment generated by gravity about the origin in space frame
        Eigen::Vector3d moment = T_sb.p.cross(mg_s);

        // Create the wrench object in space frame
        Wrench grav_s(moment, mg_s);

        // Convert the wrench to the body frame
        Wrench grav_b = grav_s.convertFrame(T_sb);

        return grav_b;
    }

    std::vector<std::tuple<double, HomoTransform, Twist, Vector6d>> solveForwardDynamic(double t0, double tf, const HomoTransform &init_T, const Twist &init_V, const std::vector<Wrench> &wrenches, double dt = 0.001)
    {
        std::vector<std::tuple<double, HomoTransform, Twist, Vector6d>> trajectory;
        HomoTransform T_sb = init_T;
        Twist V_b = init_V;

        Eigen::Matrix4d R_k = Eigen::Matrix4d::Identity();
        trajectory.emplace_back(t0, T_sb, V_b, Vector6d::Zero());

        for (double t = t0; t <= tf; t += dt)
        {
            Vector6d tau_total = gravityWrench(T_sb).Mat;
            for (const auto &wrench : wrenches)
            {
                tau_total += wrench.Mat;
            }

            Eigen::Matrix<double, 6, 6> Vq;
            Vq.block<3, 3>(0, 0) = skewSymMat(V_b.w * dt);
            Vq.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
            Vq.block<3, 3>(3, 0) = Eigen::Matrix3d::Zero();
            Vq.block<3, 3>(3, 3) = skewSymMat(V_b.w * dt);
            Vq = Vq * Mq;

            Vector6d dVdt = Mq.inverse() * (tau_total + Vq * V_b.Mat);
            // Represent body twist in current frame
            V_b.v = R_k.block<3, 3>(0, 0).transpose() * V_b.v;
            V_b.w = R_k.block<3, 3>(0, 0).transpose() * V_b.w;
            V_b = V_b.integrate(dVdt.segment<3>(3), dVdt.segment<3>(0), dt);

            // Transform base position to next in world frame
            Eigen::AngleAxisd angleAxis(V_b.w.norm() * dt, V_b.w.normalized());
            Eigen::Matrix3d R_update = angleAxis.toRotationMatrix();
            R_k.block<3, 3>(0, 0) = R_update;
            R_k.block<3, 1>(0, 3) = Eigen::Vector3d::Zero();
            Eigen::Matrix4d T_rot = T_sb.Mat * R_k;

            Eigen::Matrix4d P_ = Eigen::Matrix4d::Identity();
            P_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            P_.block<3, 1>(0, 3) = R_update * V_b.v * dt;

            Eigen::Matrix4d T_sb_next = T_sb.Mat * R_k * P_;
            T_sb.updateMat(T_sb_next.block<3, 3>(0, 0), T_sb_next.block<3, 1>(0, 3)); // Update transformation matrix

            trajectory.emplace_back(t + dt, T_sb, V_b, dVdt);
        }

        return trajectory;
    }

private:
    // Helper function to generate a skew-symmetric matrix from a vector
    static Eigen::Matrix3d skewSymMat(const Eigen::Vector3d &vx)
    {
        Eigen::Matrix3d result;
        result << 0, -vx.z(), vx.y(),
            vx.z(), 0, -vx.x(),
            -vx.y(), vx.x(), 0;
        return result;
    }

    // Computes the adjoint matrix of a twist for dynamics calculations
    Eigen::Matrix<double, 6, 6> adjointMatrix(const Twist &twist) const
    {
        Eigen::Matrix<double, 6, 6> ad;
        Eigen::Matrix3d wx = skewSymMat(twist.w); // Using previously defined skewSymMat
        Eigen::Matrix3d vx = skewSymMat(twist.v);
        ad.block<3, 3>(0, 0) = wx;
        ad.block<3, 3>(3, 3) = wx;
        ad.block<3, 3>(3, 0) = vx;
        ad.block<3, 3>(0, 3) = Eigen::Matrix3d::Zero();
        return ad;
    }
};

int main()
{
    // Initial configuration of the floating base
    Eigen::Vector3d init_position(0.0, 0.0, 0.2);                // Initial position
    Eigen::Matrix3d init_rotation = Eigen::Matrix3d::Identity(); // Initial rotation (identity for no initial rotation)
    Eigen::Vector3d init_linear_velocity(5.0, 0.0, 5.0);         // Initial linear velocity
    Eigen::Vector3d init_angular_velocity(0.0, 5.0, 0.0);        // Initial angular velocity
    // Creating initial twist
    Twist init_twist(init_linear_velocity, init_angular_velocity);

    // Instantiate FloatingBase object with initial conditions
    FloatingBase robot(init_position, init_rotation, init_twist);

    // Time parameters for the simulation
    double t0 = 0.0;   // Start time
    double tf = 5;     // End time
    double dt = 0.001; // Time step

    // Simulate forward dynamics
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Wrench> wrenches;
    std::vector<std::tuple<double, HomoTransform, Twist, Vector6d>> trajectory =
        robot.solveForwardDynamic(t0, tf, robot.T_sb, robot.V_b, wrenches, dt);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    std::cout << "Time\tX\tY\tZ" << std::endl;
    std::ofstream log("fbd.csv");
    log << "t,x,y,z\n";
    for (const auto &state : trajectory)
    {
        double t_ = std::get<0>(state);
        HomoTransform T_sb_k = std::get<1>(state);
        log << t_ << "," << T_sb_k.p[0] << "," << T_sb_k.p[1] << "," << T_sb_k.p[2] << std::endl;
        // std::cout << state[0] << "\t" << state[1] << "\t" << state[2] << "\t" << state[3] << endl;
        // log << state[0] << "," << state[1] << "," << state[2] << "," << state[3] << endl;
    }

    std::cout << "Runtime: " << duration.count() * 0.001 << " milliseconds" << std::endl;
    std::cout << "Freq: " << 1 / (duration.count() * 0.001 * 0.001) << " Hz" << std::endl;
}
