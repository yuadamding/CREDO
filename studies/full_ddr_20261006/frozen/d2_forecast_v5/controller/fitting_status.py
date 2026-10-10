"""One explicit receipt-bound fitting status snapshot; no polling or writes remotely."""
import fitting_operate
import prepare_status

if __name__ == "__main__":
    fitting_operate.bind_context()
    prepare_status.main()
