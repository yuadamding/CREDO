"""Exact read-only capacity snapshot using the tested scalar receipt adapter."""
import qualify_operate
import prepare_status


if __name__ == "__main__":
    qualify_operate.bind_context()
    prepare_status.main()
