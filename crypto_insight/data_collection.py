import trafilatura
import pdfkit

urls = [
    
   'https://thenewscrypto.com/bitcoin-is-stuck-in-limbo-bitlemons-blem-is-running-the-tables/',
   'https://thenewscrypto.com/altcoins-and-meme-coins-drop-despite-trumps-bitcoin-reserve-announcement/',
   'https://thenewscrypto.com/4-bullish-reasons-why-trumps-bitcoin-reserve-is-a-game-changer/',
   'https://thenewscrypto.com/500-to-714k-arctic-pablos-insane-prediction-goes-viral-while-dog-bitcoin-and-sudeng-eye-innovation-best-meme-coins-to-buy-this-week/',
   'https://thenewscrypto.com/white-house-establishes-strategic-bitcoin-reserve-ahead-of-crypto-summit/',
   'https://thenewscrypto.com/bitcoin-slides-to-87k-despite-crypto-summit-and-strategic-reserve-execution/',
   'https://thenewscrypto.com/is-bitcoin-price-going-to-crash-with-the-crypto-summit-coming-up/',
   'https://thenewscrypto.com/michael-saylor-advocates-for-a-u-s-bitcoin-reserve-at-white-house-crypto-summit/',
   'https://thenewscrypto.com/bitcoin-cash-bch-price-prediction/',
   'https://thenewscrypto.com/new-hampshire-bitcoin-bill-moves-closer-to-final-house-vote/',
   'https://thenewscrypto.com/bitcoin-cash-bch-surges-over-30-this-week-will-it-sustain/',
   'https://thenewscrypto.com/bitwise-claims-trumps-bitcoin-reserve-will-be-larger-than-expected/',
   'https://thenewscrypto.com/mt-gox-transfers-12000-btc-worth-over-1b-as-bitcoin-regained-92k/',
   'https://thenewscrypto.com/best-crypto-investment-qubetics-ide-opens-blockchain-to-all-xrp-faces-sec-appeal-and-bitcoins-u-s-reserve-potential-sparks-debate/',
   'https://thenewscrypto.com/learnbitcoin-com-launches-personalized-be-your-own-bank-platform-to-simplify-bitcoin-ownership/',
   'https://thenewscrypto.com/metaplanet-stock-surges-20-after-purchasing-497-additional-bitcoins/',
   'https://thenewscrypto.com/mexican-billionaire-ricardo-salinas-allocates-70-of-portfolio-to-bitcoin/',
   'https://thenewscrypto.com/el-salvador-refuses-to-halt-bitcoin-accumulation-under-imf-deal/',
   'https://thenewscrypto.com/bitcoin-shows-signs-of-revival-will-global-economic-dispute-allow-bull-run/',
   'https://thenewscrypto.com/bitcoin-btc-drops-below-83k-as-crypto-market-sees-sharp-decline/'

]

articles_content = ""

for url in urls:
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        content = trafilatura.extract(downloaded)
        if content:
            articles_content += f"<p>{content}</p>"

if articles_content:
    with open('articles.html', 'w', encoding='utf-8') as file:
        file.write(articles_content)
    pdfkit.from_file('articles.html', 'Crypto_Articles1.pdf')
    print("PDF Created")