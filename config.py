class Config:
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    TICKET_SUMMARY_TRANSLATED = "ticket_summary_translated"
    INTERACTION_CONTENT_TRANSLATED = "interaction_content_translated"

    TEXT_FEATURES = [TICKET_SUMMARY_TRANSLATED, INTERACTION_CONTENT_TRANSLATED]

    RAW_DATA_PATH = "data/raw/"
    PROCESSED_CSV_PATH = "data/processed/cleaned_data.csv"

    # FLAG DE CONTROLE: If set to True, forces a clean sweep, ignoring the saved file.
    #It's vital to leave this set to True in your first running
    FORCE_REPROCESS = False

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'

