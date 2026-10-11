//! Binary `histo3d`: pembungkus tipis di atas pustaka `h3d_app`.

fn main() -> std::process::ExitCode {
    env_logger::init();
    let args = match h3d_app::args::CliArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{msg}");
            eprintln!("{}", h3d_app::args::USAGE);
            return std::process::ExitCode::from(1);
        }
    };
    if args.help {
        println!("{}", h3d_app::args::USAGE);
        return std::process::ExitCode::SUCCESS;
    }
    match h3d_app::run(args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("histo3d: {e}");
            std::process::ExitCode::from(2)
        }
    }
}
