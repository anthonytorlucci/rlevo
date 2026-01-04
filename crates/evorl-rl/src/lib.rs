pub mod algorithms {
    //! Machine learning algorithms for RL.

    pub mod dqn {
        //! Deep Q-Network algorithm.

        pub mod dqn_agent;
        pub mod dqn_config;
        pub mod dqn_model;
    }
}

pub mod utils;
