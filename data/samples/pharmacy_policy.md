# Store Pharmacy Operations Policy (Demo)

This is fictional sample content for the AI Friday workshop.

## 1. Returns and refunds for prescriptions

Dispensed prescription medications cannot be returned or refunded once they leave the pharmacy counter, for patient-safety reasons.

The only exception is a dispensing error (wrong drug, wrong strength, or wrong patient). A dispensing error must be reported within 7 days of pickup. The pharmacist on duty logs it in the Rx Dispensing System and issues a full refund. The returned medication is quarantined and never restocked.

Over-the-counter items may be returned within 30 days with a receipt, if the packaging is unopened.

## 2. Controlled substances

Controlled substances (Schedule II to V) are stored in the locked cabinet. Only a licensed pharmacist may open it. Every opening is recorded in the controlled-substance log with the time, the pharmacist's name and the reason.

A count discrepancy of any size must be reported to the pharmacy manager the same day, and a ticket filed with priority High.

## 3. Label printer and POS issues

If a pharmacy label printer stops printing, first reseat the label roll and close the lid firmly. If the printer still shows offline, check its status in the IT system and file a ticket. Never hand-write a prescription label.

If a POS terminal freezes at checkout, hold the power button for 10 seconds to restart it. If it freezes more than twice in one shift, file a ticket with priority High.

## 4. Account access

Pharmacy system accounts are personal. Never share a login. Password resets are sent as a one-time link to the employee's email on file and expire after 15 minutes. Role changes, for example from store associate to pharmacist, must be approved by an IT admin and are recorded in the audit log.

## 5. Roles and what each may do

There are three roles. A store associate serves customers, checks stock and system status, and files IT tickets. A store associate may not open the controlled-substance cabinet, dispense medication or change anyone's account.

A pharmacist may do everything a store associate does, and also dispenses prescriptions, opens the controlled-substance cabinet and logs dispensing errors.

An IT admin manages user accounts and roles, sends password reset links and resets test data. An IT admin never dispenses medication.

## 6. Low stock and out-of-stock items

If a product has fewer than 20 units at a store, file a ticket with priority Medium to request a restock. Always check live stock in the inventory system first; never quote a number from memory.

If a product is out of stock (0 units), check the partner store before telling the customer. Store 1042 and store 2201 are partner stores. If the partner store has stock, offer the customer a transfer.

## 7. Choosing a ticket priority

High: patient safety is at risk, or sales or dispensing have stopped. Examples: a controlled-substance count discrepancy, a POS terminal that freezes more than twice in one shift, or the Rx Dispensing System being down.

Medium: a workaround exists. Examples: a label printer that is offline while another printer works, or a low-stock restock request.

Low: cosmetic or minor issues, such as a faded shelf label or a slow but working Wi-Fi connection.

## 8. Wi-Fi and network outages

If the store Wi-Fi is down, POS terminals switch to offline mode and can take card payments for up to 4 hours. The Rx Dispensing System must not be used offline; pause dispensing and file a ticket with priority High.

## 9. Resetting test data

Only an IT admin may reset data, and only in demo or test environments. Always run a preview (dry run) first and show the row counts. A second person must approve the reset before it runs, and every reset is recorded in the audit log.

## 10. IT service desk

For anything the IT assistant cannot resolve, contact the IT service desk at servicedesk@example.com or on extension 4400. The service desk is staffed from 7:00 to 22:00, seven days a week. Outside those hours, file a ticket and it is picked up the next morning.
