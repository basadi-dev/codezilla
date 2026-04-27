mod commands;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: calc <command> <a> <b>");
        eprintln!("Commands: add, multiply, divide");
        std::process::exit(1);
    }

    let command = &args[1];
    let a: f64 = args[2].parse().expect("invalid number");
    let b: f64 = args[3].parse().expect("invalid number");

    let result = match command.as_str() {
        "add" => Ok(commands::add::add(a, b)),
        "multiply" => Ok(commands::multiply::multiply(a, b)),
        // TODO: add "divide" command here
        _ => {
            eprintln!("Unknown command: {command}");
            std::process::exit(1);
        }
    };

    match result {
        Ok(value) => println!("{value}"),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
